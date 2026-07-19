//
//  CameraManager.swift
//  ClimbPulse
//
//  Manages AVCaptureSession for PPG recording using the back camera with torch.
//  Extracts average red channel values from each frame for heart rate estimation.
//

import Foundation
@preconcurrency import AVFoundation
import UIKit
import Combine

/// A selectable rear camera. `id` is the device's stable `uniqueID` so the
/// user's choice can be persisted and resolved again later.
struct CameraOption: Identifiable, Hashable, Sendable {
    let id: String
    let name: String
    /// Compact label (e.g. "0.5×") for space-constrained controls like the
    /// segmented selector on the recording screen.
    let shortName: String
}

/// Thread-safe storage for state accessed from multiple threads (recording
/// timing plus the currently-selected camera, which the capture/torch code
/// running on the session queue needs to read).
/// `nonisolated` opts out of the project's default main-actor isolation so the
/// lock-guarded state can be read/written from the capture delegate's background queue.
nonisolated final class RecordingState: @unchecked Sendable {
    private let lock = NSLock()
    private var _startTime: Date?
    private var _startPTS: CMTime?
    private var _cameraID: String?

    var startTime: Date? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _startTime
        }
        set {
            lock.lock()
            _startTime = newValue
            lock.unlock()
        }
    }

    var startPTS: CMTime? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _startPTS
        }
        set {
            lock.lock()
            _startPTS = newValue
            lock.unlock()
        }
    }

    /// `uniqueID` of the camera to capture with; `nil` falls back to the default.
    var cameraID: String? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _cameraID
        }
        set {
            lock.lock()
            _cameraID = newValue
            lock.unlock()
        }
    }
}

/// Manages camera capture and PPG signal extraction from video frames.
@MainActor
class CameraManager: NSObject, ObservableObject {
    
    // MARK: - Published Properties
    
    @Published var isRecording = false
    @Published var currentBPM: Int?
    @Published var samples: [PPGSample] = []
    @Published var filteredSamples: [PPGSample] = []  // Band-passed samples for UI display
    @Published var timeRemaining: Int = 60
    @Published var errorMessage: String?
    @Published var isAuthorized = false
    @Published private(set) var captureSession: AVCaptureSession?
    @Published var signalQuality: SignalQuality = .noisy

    /// All rear cameras the user can pick between (populated on authorization).
    @Published private(set) var availableCameras: [CameraOption] = []

    /// `uniqueID` of the camera to record with. Persisted so a chosen lens
    /// sticks across launches while experimenting with which gives the best
    /// torch illumination. `nil` means "use the system default".
    @Published var selectedCameraID: String? {
        didSet {
            UserDefaults.standard.set(selectedCameraID, forKey: Self.selectedCameraKey)
        }
    }

    var recordingLength: Int { recordingDuration }

    // MARK: - Camera Discovery

    private static let selectedCameraKey = "climbpulse_selected_camera"

    /// Rear physical lenses we let the user choose between. Ordered so the lens
    /// physically closest to the LED torch (ultra-wide, on Pro models) comes first.
    nonisolated private static let selectableDeviceTypes: [AVCaptureDevice.DeviceType] = [
        .builtInUltraWideCamera,
        .builtInWideAngleCamera,
        .builtInTelephotoCamera
    ]

    // MARK: - Private Properties
    
    private var videoOutput: AVCaptureVideoDataOutput?
    private let sessionQueue = DispatchQueue(label: "com.climbpulse.camera.session")
    private let processingQueue = DispatchQueue(label: "com.climbpulse.camera.processing")
    private var previousRedValue: Double?
    
    // Thread-safe recording state
    private let recordingState = RecordingState()
    
    private var timer: Timer?
    private let recordingDuration: Int = 60

    // Throttle for the torch keep-alive check (MainActor-isolated).
    private var lastTorchCheck: Date = .distantPast
    
    private let ppgProcessor = PPGProcessor()
    private var lastBPMUpdate: Date = Date()
    private let bpmUpdateInterval: TimeInterval = 2.0  // Update BPM every 2 seconds
    private var detectionStartTimestamp: Double?
    
    // Start countdown only after BPM detected
    private var countdownStarted = false
    
    // Completion handler for when recording finishes
    var onRecordingComplete: ((Measurement?) -> Void)?
    
    // MARK: - Setup

    override init() {
        selectedCameraID = UserDefaults.standard.string(forKey: Self.selectedCameraKey)
        super.init()
    }

    /// Request camera authorization.
    func requestAuthorization() async {
        let status = AVCaptureDevice.authorizationStatus(for: .video)

        switch status {
        case .authorized:
            isAuthorized = true
        case .notDetermined:
            isAuthorized = await AVCaptureDevice.requestAccess(for: .video)
        case .denied, .restricted:
            isAuthorized = false
            errorMessage = "Camera access denied. Please enable in Settings."
        @unknown default:
            isAuthorized = false
        }

        if isAuthorized {
            refreshAvailableCameras()
        }
    }

    /// Enumerate the rear cameras and expose them for selection. Called after
    /// authorization is granted (camera metadata is only available then).
    func refreshAvailableCameras() {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: Self.selectableDeviceTypes,
            mediaType: .video,
            position: .back
        )

        // Present the lenses in a stable, meaningful order (ultra-wide, wide,
        // telephoto) rather than trusting the discovery array's ordering.
        let ordered = discovery.devices.sorted {
            (Self.selectableDeviceTypes.firstIndex(of: $0.deviceType) ?? .max)
                < (Self.selectableDeviceTypes.firstIndex(of: $1.deviceType) ?? .max)
        }
        availableCameras = ordered.map { device in
            CameraOption(
                id: device.uniqueID,
                name: Self.displayName(for: device),
                shortName: Self.shortDisplayName(for: device)
            )
        }

        // If nothing is selected yet (or the saved lens is gone), fall back to the
        // preferred default lens — ultra-wide, then wide, never telephoto.
        if selectedCameraID == nil || !availableCameras.contains(where: { $0.id == selectedCameraID }) {
            selectedCameraID = Self.preferredDefaultCamera(from: discovery.devices)?.uniqueID
                ?? availableCameras.first?.id
        }
    }

    /// Preferred default lens for PPG: ultra-wide (closest to the torch, shortest
    /// minimum focus distance), then wide. Deliberately never telephoto (far from
    /// the torch, can't focus on a pressed fingertip). Virtual multi-lens devices
    /// are excluded upstream at the `DiscoverySession` level so iOS can't silently
    /// switch physical lenses mid-recording and inject a step artifact.
    nonisolated private static func preferredDefaultCamera(from devices: [AVCaptureDevice]) -> AVCaptureDevice? {
        devices.first { $0.deviceType == .builtInUltraWideCamera }
            ?? devices.first { $0.deviceType == .builtInWideAngleCamera }
    }

    /// Human-friendly label for a lens, e.g. "Ultra-Wide (0.5×)".
    nonisolated private static func displayName(for device: AVCaptureDevice) -> String {
        switch device.deviceType {
        case .builtInUltraWideCamera:
            return "Ultra-Wide (0.5×)"
        case .builtInWideAngleCamera:
            return "Wide (1×)"
        case .builtInTelephotoCamera:
            return "Telephoto"
        default:
            return device.localizedName
        }
    }

    /// Compact label for a lens, e.g. "0.5×", for the segmented selector.
    nonisolated private static func shortDisplayName(for device: AVCaptureDevice) -> String {
        switch device.deviceType {
        case .builtInUltraWideCamera:
            return "0.5×"
        case .builtInWideAngleCamera:
            return "1×"
        case .builtInTelephotoCamera:
            return "Tele"
        default:
            return device.localizedName
        }
    }

    /// Resolve the `AVCaptureDevice` to use, preferring the given `uniqueID` and
    /// falling back to the wide-angle lens (then any rear camera) if it's gone.
    /// Shared by capture setup and torch control so they always agree.
    nonisolated private static func resolveCamera(id: String?) -> AVCaptureDevice? {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: selectableDeviceTypes,
            mediaType: .video,
            position: .back
        )
        if let id, let match = discovery.devices.first(where: { $0.uniqueID == id }) {
            return match
        }
        return preferredDefaultCamera(from: discovery.devices)
            ?? discovery.devices.first
            ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
    }
    
    /// Set up the capture session with back camera and torch.
    nonisolated private func setupCaptureSession() throws -> (AVCaptureSession, AVCaptureVideoDataOutput) {
        let session = AVCaptureSession()
        session.sessionPreset = .vga640x480  // Small frame, good throughput

        // Get the user-selected back camera (falls back to the wide-angle lens).
        guard let camera = Self.resolveCamera(id: recordingState.cameraID) else {
            throw CameraError.cameraUnavailable
        }
        
        // Configure camera for PPG capture
        try camera.lockForConfiguration()
        
        // Lock exposure and white balance for consistent readings
        if camera.isExposureModeSupported(.locked) {
            camera.exposureMode = .locked
        }
        if camera.isWhiteBalanceModeSupported(.locked) {
            camera.whiteBalanceMode = .locked
        }
        
        // Set frame rate to 30 fps for good temporal resolution
        camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
        camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
        
        camera.unlockForConfiguration()
        
        // Add camera input
        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else {
            throw CameraError.cannotAddInput
        }
        session.addInput(input)
        
        // Set up video output
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: processingQueue)
        output.alwaysDiscardsLateVideoFrames = false  // keep frames to reach target rate
        
        guard session.canAddOutput(output) else {
            throw CameraError.cannotAddOutput
        }
        session.addOutput(output)
        
        return (session, output)
    }
    
    // MARK: - Recording Control
    
    /// Start PPG recording session.
    func startRecording() {
        guard isAuthorized else {
            errorMessage = "Camera not authorized"
            return
        }
        
        // Reset state
        samples = []
        filteredSamples = []
        currentBPM = nil
        timeRemaining = recordingDuration
        errorMessage = nil
        recordingState.startTime = nil
        recordingState.startPTS = nil
        captureSession = nil
        signalQuality = .noisy
        countdownStarted = false
        previousRedValue = nil
        detectionStartTimestamp = nil

        // Hand the chosen camera to the session-queue code in a thread-safe way.
        recordingState.cameraID = selectedCameraID

        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            do {
                let (session, output) = try self.setupCaptureSession()

                // Set recording start time before starting session
                self.recordingState.startTime = Date()

                // Start capture session
                session.startRunning()

                // Enable torch (flash) for illumination.
                // Must be done AFTER the session is running, otherwise starting
                // the session resets the torch mode and the flash never turns on.
                try self.setTorch(on: true)
                
                Task { @MainActor in
                    self.captureSession = session
                    self.videoOutput = output
                    self.isRecording = true
                    // NOTE: Do NOT start the timer here.
                    // We only start the countdown after BPM is detected.
                }
            } catch {
                Task { @MainActor in
                    self.errorMessage = "Failed to start camera: \(error.localizedDescription)"
                }
            }
        }
    }
    
    /// Stop recording and process final results.
    func stopRecording() {
        timer?.invalidate()
        timer = nil
        
        let session = captureSession
        
        sessionQueue.async { [weak self] in
            guard let self = self else { return }

            do {
                try self.setTorch(on: false)
            } catch {
                Task { @MainActor in
                    self.errorMessage = "Failed to turn off flash: \(error.localizedDescription)"
                }
            }
            session?.stopRunning()
            self.recordingState.startTime = nil
            self.recordingState.startPTS = nil
            
            Task { @MainActor in
                self.captureSession = nil
                self.isRecording = false
                self.processAndSaveMeasurement()
            }
        }
    }
    
    /// Switch to a different rear lens. If a recording is in progress the live
    /// capture session is torn down and rebuilt with the new camera (the sample
    /// buffer and countdown restart from scratch, since the two lenses' signals
    /// aren't comparable). Outside of a recording this just updates the stored
    /// preference, exactly like the selector on the home screen.
    func switchCamera(to id: String?) {
        guard id != selectedCameraID else { return }
        selectedCameraID = id

        guard isRecording else { return }

        // Stop the countdown; startRecording() will restart it once BPM is
        // re-detected on the new lens.
        timer?.invalidate()
        timer = nil

        // Stop the old session first. sessionQueue is serial, so this is
        // guaranteed to run before startRecording()'s setup below. Stopping the
        // session also releases the torch, which startRecording() re-enables.
        let oldSession = captureSession
        sessionQueue.async {
            oldSession?.stopRunning()
        }

        // Rebuild the session with the freshly selected camera. This resets all
        // sample/BPM/quality state, so no stale data from the old lens leaks in.
        startRecording()
    }

    // MARK: - Private Methods

    /// Enable/disable camera torch.
    nonisolated private func setTorch(on: Bool) throws {
        guard let device = Self.resolveCamera(id: recordingState.cameraID),
              device.hasTorch, device.isTorchAvailable else { return }

        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            if on {
                let desiredLevel = min(0.6, AVCaptureDevice.maxAvailableTorchLevel)
                try device.setTorchModeOn(level: desiredLevel)  // Slightly dim to avoid sensor saturation
            } else {
                device.torchMode = .off
            }
        } catch {
            throw CameraError.torchUnavailable(underlying: error)
        }
    }

    /// Re-assert the torch if something (session reconfiguration, a transient
    /// interruption) has switched it back off while we are still recording.
    nonisolated private func ensureTorchOn() throws {
        guard let device = Self.resolveCamera(id: recordingState.cameraID),
              device.hasTorch, device.isTorchAvailable, !device.isTorchActive else { return }
        try setTorch(on: true)
    }
    
    /// Start countdown timer.
    private func startTimer() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            
            Task { @MainActor in
                if self.timeRemaining > 0 {
                    self.timeRemaining -= 1
                } else {
                    self.stopRecording()
                }
            }
        }
    }
    
    /// Calculate estimated sample rate from collected samples.
    private func estimatedSampleRate() -> Double {
        guard samples.count > 1,
              let first = samples.first?.timestamp,
              let last = samples.last?.timestamp else {
            return 30.0  // Default assumption
        }
        
        let duration = last - first
        guard duration > 0 else { return 30.0 }
        
        return Double(samples.count) / duration
    }
    
    /// Process collected samples and create measurement.
    private func processAndSaveMeasurement() {
        guard samples.count > 50 else {
            errorMessage = "Insufficient data collected"
            onRecordingComplete?(nil)
            return
        }
        
        let sampleRate = estimatedSampleRate()
        
        // Trim to post-detection window if available to avoid early noisy jump
        let trimmed: [PPGSample]
        if let startTs = detectionStartTimestamp {
            trimmed = samples.filter { $0.timestamp >= startTs }
        } else {
            trimmed = samples
        }
        
        guard trimmed.count > 20 else {
            errorMessage = "Insufficient stable data collected"
            onRecordingComplete?(nil)
            return
        }
        
        // Clean signal for BPM/quality/storage
        let cleaned = ppgProcessor.cleanedSignal(samples: trimmed, sampleRate: sampleRate)
        
        // Calculate final BPM
        let finalBPM = ppgProcessor.calculateBPM(from: cleaned, sampleRate: sampleRate) ?? currentBPM ?? 0
        
        // Assess quality
        let quality = ppgProcessor.assessQuality(samples: cleaned, sampleRate: sampleRate)
        
        // Downsample for storage (use cleaned values)
        let downsampledPPG = ppgProcessor.downsample(samples: cleaned)
        
        // Calculate actual duration
        let duration = cleaned.last!.timestamp - cleaned.first!.timestamp
        
        // Create measurement
        let measurement = Measurement(
            userId: Self.getUserId(),
            duration: duration,
            sampleRate: sampleRate,
            bpm: finalBPM,
            quality: quality,
            ppgData: downsampledPPG
        )
        
        onRecordingComplete?(measurement)
    }
    
    /// Get or create anonymous user ID.
    nonisolated private static func getUserId() -> String {
        let key = "climbpulse_user_id"
        if let existingId = UserDefaults.standard.string(forKey: key) {
            return existingId
        }
        let newId = UUID().uuidString
        UserDefaults.standard.set(newId, forKey: key)
        return newId
    }
    
    /// Update BPM estimate periodically.
    private func updateBPMIfNeeded() {
        let now = Date()
        guard now.timeIntervalSince(lastBPMUpdate) >= bpmUpdateInterval else { return }
        lastBPMUpdate = now
        
        let sampleRate = estimatedSampleRate()
        
        // Compute BPM
        let bpm = ppgProcessor.calculateBPM(from: samples, sampleRate: sampleRate)
        self.currentBPM = bpm
        
        // Refresh signal quality alongside BPM updates
        if !samples.isEmpty {
            self.signalQuality = ppgProcessor.assessQuality(samples: samples, sampleRate: sampleRate)
        }
        
        // Start countdown only when BPM is first detected.
        if !countdownStarted, bpm != nil {
            beginCountdownFromNow()
        }
    }
    
    /// Rebase timestamps and storage so we only keep data from the moment BPM was detected,
    /// and start the countdown from that moment.
    private func beginCountdownFromNow() {
        countdownStarted = true
        
        // Start countdown without clearing existing samples to avoid visible jumps
        self.recordingState.startTime = Date()
        self.detectionStartTimestamp = samples.last?.timestamp
        self.timeRemaining = recordingDuration
        self.startTimer()
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    /// Process each video frame to extract PPG signal.
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        // Lazily set the first presentation timestamp to align future frames
        if recordingState.startPTS == nil {
            recordingState.startPTS = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        }
        guard let startPTS = recordingState.startPTS else { return }
        
        // Extract PPG signal from frame (now using green channel from ROI)
        guard let ppgValue = extractPPGSignal(from: sampleBuffer) else { return }
        // Temporarily remove frame-level filtering to debug signal flow
        // guard ppgValue > 5 && ppgValue < 250 else { return }  // Skip too dark/saturated frames
        
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let relative = CMTimeSubtract(pts, startPTS)
        let timestamp = CMTimeGetSeconds(relative)
        
        Task { @MainActor in
            self.appendSample(value: ppgValue, timestamp: timestamp)
        }
    }
    
    /// Append a smoothed PPG sample on the main actor to avoid cross-thread state mutation.
    @MainActor
    private func appendSample(value: Double, timestamp: Double) {
        let smoothed = smoothPPGValue(value)
        let limited = clampJump(smoothed)
        let sample = PPGSample(timestamp: timestamp, value: limited)
        samples.append(sample)
        
        // Refresh filtered copy for UI and quality estimate
        let sampleRate = estimatedSampleRate()
        filteredSamples = ppgProcessor.filteredForDisplay(
            samples: samples,
            sampleRate: sampleRate
        )
        updateBPMIfNeeded()
        maybeReassertTorch()
    }

    /// Re-assert the torch at most once per second while recording, in case a
    /// session reconfiguration or transient interruption switched it back off.
    /// Hardware access is hopped onto the session queue to stay off the main thread.
    @MainActor
    private func maybeReassertTorch() {
        guard isRecording else { return }
        let now = Date()
        guard now.timeIntervalSince(lastTorchCheck) >= 1.0 else { return }
        lastTorchCheck = now

        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                try self.ensureTorchOn()
            } catch {
                Task { @MainActor in
                    self.errorMessage = "Flash error: \(error.localizedDescription)"
                }
            }
        }
    }
    
    /// Exponential smoothing for PPG values to suppress frame-to-frame noise.
    @MainActor
    private func smoothPPGValue(_ newValue: Double) -> Double {
        let clamped = min(max(newValue, 0), 255)
        
        guard let previous = previousRedValue else {
            previousRedValue = clamped
            return clamped
        }
        
        // Heavier smoothing on sudden jumps (likely motion noise)
        let jump = abs(clamped - previous)
        let alpha: Double = jump > 60 ? 0.18 : 0.32
        let blended = previous * (1 - alpha) + clamped * alpha
        previousRedValue = blended
        return blended
    }
    
    /// Cap abrupt jumps between consecutive samples to reduce motion spikes.
    @MainActor
    private func clampJump(_ value: Double, maxDelta: Double = 18.0) -> Double {
        guard let prev = samples.last?.value else {
            return value
        }
        let delta = value - prev
        if abs(delta) <= maxDelta {
            return value
        }
        return prev + (delta > 0 ? maxDelta : -maxDelta)
    }
    
    /// Extract PPG signal from centered ROI using green channel for improved SNR.
    /// Uses centered ROI to avoid edge glare/vignetting, filters out saturated/dim pixels,
    /// and uses green channel which typically provides better PPG signal than red.
    nonisolated private func extractPPGSignal(from sampleBuffer: CMSampleBuffer) -> Double? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)

        var greenSum: Int = 0
        var validPixelCount: Int = 0

        // BGRA format: B=0, G=1, R=2, A=3
        // Sample the full frame with light subsampling to reduce noise while
        // keeping computation low on the low-resolution buffer.
        let step = 2  // every other pixel is enough at .low preset
        for y in stride(from: 0, to: height, by: step) {
            let rowStart = y * bytesPerRow
            for x in stride(from: 0, to: width, by: step) {
                let offset = rowStart + x * 4
                let green = Int(buffer[offset + 1])  // Green channel

                greenSum += green
                validPixelCount += 1
            }
        }

        guard validPixelCount > 0 else { return nil }  // Need at least one pixel

        let avgGreen = Double(greenSum) / Double(validPixelCount)

        // Green channel typically has better SNR for PPG than red.
        return avgGreen
    }
}

// MARK: - Errors

enum CameraError: LocalizedError {
    case cameraUnavailable
    case cannotAddInput
    case cannotAddOutput
    case torchUnavailable(underlying: Error)

    var errorDescription: String? {
        switch self {
        case .cameraUnavailable:
            return "Back camera is not available"
        case .cannotAddInput:
            return "Cannot add camera input"
        case .cannotAddOutput:
            return "Cannot add video output"
        case .torchUnavailable(let underlying):
            return "Could not control the camera flash: \(underlying.localizedDescription)"
        }
    }
}
