//
//  CameraManager.swift
//  ClimbPulse
//
//  Manages AVCaptureSession for PPG recording using the back camera with torch.
//  Extracts an averaged colour-channel value from each frame for heart rate
//  estimation. Capture options (channel, ROI, pixel filtering, torch level) are
//  user-adjustable via CaptureTuning.
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

/// Colour channel averaged from each frame to form the PPG signal.
/// `nonisolated` so the capture delegate (running off the main actor) can read it.
nonisolated enum PPGChannel: String, CaseIterable, Identifiable, Sendable {
    case red
    case green

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .red: return "Red"
        case .green: return "Green"
        }
    }
}

/// Snapshot of the user-adjustable capture options, read from the capture and
/// torch code on their background queues. A value type so it can be copied out
/// from under a lock without tearing. `nonisolated` so the capture/torch code
/// off the main actor can read it.
nonisolated struct CaptureTuning: Sendable, Equatable {
    var channel: PPGChannel = .red
    var useROI: Bool = true
    var filterPixels: Bool = true
    var torchLevel: Double = 0.3
    /// Minimum red-to-blue ratio for a frame to count as "finger on the lens".
    /// Partial coverage lets ambient light in and lowers this ratio, so a higher
    /// value demands fuller coverage before measuring.
    var minRedBlueRatio: Double = 2.0
}

/// Per-frame colour averages plus the selected-channel PPG value. Used both to
/// build the PPG signal and to decide whether a fingertip is actually covering
/// the lens. `nonisolated` so the capture delegate can produce it off the main
/// actor.
nonisolated struct FrameStats: Sendable {
    let ppgValue: Double   // averaged selected channel (red or green)
    let meanRed: Double
    let meanGreen: Double
    let meanBlue: Double
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

    private var _tuning = CaptureTuning()

    /// User-adjustable capture options, read per-frame by the capture delegate
    /// and by the torch code.
    var tuning: CaptureTuning {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _tuning
        }
        set {
            lock.lock()
            _tuning = newValue
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
    @Published var errorMessage: String?
    @Published var isAuthorized = false
    @Published private(set) var captureSession: AVCaptureSession?
    @Published var signalQuality: SignalQuality = .noisy

    /// Whether a fingertip currently appears to be covering the lens (frame is
    /// mostly red/orange). Drives the "place your finger" prompt and gates which
    /// frames contribute to the signal.
    @Published private(set) var fingerDetected: Bool = false

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

    // MARK: - Signal Tuning (user-adjustable capture options)

    /// Colour channel averaged for the PPG signal. Red has by far the best SNR
    /// for transmission through a fingertip pressed over the torch; green is
    /// offered for comparison / reflectance-style placement.
    @Published var ppgChannel: PPGChannel = .red {
        didSet {
            UserDefaults.standard.set(ppgChannel.rawValue, forKey: Self.channelKey)
            syncTuning()
        }
    }

    /// Average only a centered region of the frame to avoid edge glare and
    /// lens vignetting.
    @Published var useROI: Bool = true {
        didSet {
            UserDefaults.standard.set(useROI, forKey: Self.roiKey)
            syncTuning()
        }
    }

    /// Drop under-exposed / saturated pixels, which carry no pulse information.
    @Published var filterPixels: Bool = true {
        didSet {
            UserDefaults.standard.set(filterPixels, forKey: Self.pixelFilterKey)
            syncTuning()
        }
    }

    /// Torch brightness. Kept moderate (default 0.3, capped at 0.6) so the LED
    /// stays cool and doesn't push the sensor into saturation.
    @Published var torchLevel: Double = CameraManager.defaultTorchLevel {
        didSet {
            UserDefaults.standard.set(torchLevel, forKey: Self.torchLevelKey)
            syncTuning()
            reapplyTorchIfRecording()
        }
    }

    /// How fully the fingertip must cover the lens before frames count (higher =
    /// stricter). Expressed as the minimum red-to-blue ratio of the frame.
    @Published var fingerCoverageStrictness: Double = CameraManager.defaultFingerStrictness {
        didSet {
            UserDefaults.standard.set(fingerCoverageStrictness, forKey: Self.fingerStrictnessKey)
            syncTuning()
        }
    }

    static let defaultTorchLevel: Double = 0.3
    static let minUserTorchLevel: Double = 0.1
    static let maxUserTorchLevel: Double = 0.6

    static let defaultFingerStrictness: Double = 2.0
    static let minFingerStrictness: Double = 1.2
    static let maxFingerStrictness: Double = 3.5

    // MARK: - Camera Discovery

    private static let selectedCameraKey = "climbpulse_selected_camera"
    private static let channelKey = "climbpulse_ppg_channel"
    private static let roiKey = "climbpulse_use_roi"
    private static let pixelFilterKey = "climbpulse_filter_pixels"
    private static let torchLevelKey = "climbpulse_torch_level"
    private static let fingerStrictnessKey = "climbpulse_finger_strictness"

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
    private var previousPPGValue: Double?

    // Thread-safe recording state
    private let recordingState = RecordingState()

    // Throttle for the torch keep-alive check (MainActor-isolated).
    private var lastTorchCheck: Date = .distantPast

    private let ppgProcessor = PPGProcessor()
    private var lastBPMUpdate: Date = Date()
    private let bpmUpdateInterval: TimeInterval = 2.0  // Update BPM every 2 seconds
    private var detectionStartTimestamp: Double?

    /// Trailing window (seconds) used for the live BPM readout. Kept short so the
    /// displayed value tracks the current heartbeat instead of averaging over the
    /// whole (now open-ended) recording.
    private let liveBPMWindowSeconds: Double = 6.0

    /// Cap on how much recent signal we retain while recording. The measurement
    /// now runs indefinitely, so we keep only a rolling window to bound memory and
    /// per-frame processing cost; everything the UI and BPM estimate need lives
    /// well within this span.
    private let maxRetainedSeconds: Double = 20.0

    // Only mark the detection start (to trim the noisy lead-in) once BPM appears.
    private var detectionStarted = false
    
    // Completion handler for when recording finishes
    var onRecordingComplete: ((Measurement?) -> Void)?
    
    // MARK: - Setup

    override init() {
        selectedCameraID = UserDefaults.standard.string(forKey: Self.selectedCameraKey)
        super.init()
        loadTuningFromDefaults()
    }

    /// Restore persisted tuning options (falling back to defaults) and seed the
    /// thread-safe snapshot the capture code reads.
    private func loadTuningFromDefaults() {
        let defaults = UserDefaults.standard
        if let raw = defaults.string(forKey: Self.channelKey),
           let channel = PPGChannel(rawValue: raw) {
            ppgChannel = channel
        }
        if defaults.object(forKey: Self.roiKey) != nil {
            useROI = defaults.bool(forKey: Self.roiKey)
        }
        if defaults.object(forKey: Self.pixelFilterKey) != nil {
            filterPixels = defaults.bool(forKey: Self.pixelFilterKey)
        }
        if defaults.object(forKey: Self.torchLevelKey) != nil {
            torchLevel = min(Self.maxUserTorchLevel,
                             max(Self.minUserTorchLevel, defaults.double(forKey: Self.torchLevelKey)))
        }
        if defaults.object(forKey: Self.fingerStrictnessKey) != nil {
            fingerCoverageStrictness = min(Self.maxFingerStrictness,
                                           max(Self.minFingerStrictness, defaults.double(forKey: Self.fingerStrictnessKey)))
        }
        syncTuning()  // ensure the snapshot is populated even if nothing was persisted
    }

    /// Push the current user-adjustable options into the lock-guarded snapshot
    /// the capture/torch code reads from its background queues.
    private func syncTuning() {
        recordingState.tuning = CaptureTuning(
            channel: ppgChannel,
            useROI: useROI,
            filterPixels: filterPixels,
            torchLevel: torchLevel,
            minRedBlueRatio: fingerCoverageStrictness
        )
    }

    /// Re-apply the torch when its level changes mid-measurement so the new
    /// brightness takes effect immediately.
    private func reapplyTorchIfRecording() {
        guard isRecording else { return }
        sessionQueue.async { [weak self] in
            guard let self else { return }
            do {
                try self.setTorch(on: true)
            } catch {
                Task { @MainActor in
                    self.errorMessage = "Flash error: \(error.localizedDescription)"
                }
            }
        }
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

        // Start in continuous auto exposure / white balance so the sensor can
        // settle on the finger+torch scene; both are locked shortly after the
        // torch comes on (see lockExposureAndWhiteBalance). Locking here — before
        // the torch and finger are in place — would freeze a dark-scene exposure
        // that then actively fights the pulse waveform.
        if camera.isExposureModeSupported(.continuousAutoExposure) {
            camera.exposureMode = .continuousAutoExposure
        }
        if camera.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
            camera.whiteBalanceMode = .continuousAutoWhiteBalance
        }

        // Fixed 30 fps: plenty for heart-rate timing, and a constant frame
        // duration keeps the signal from being resampled by auto frame-rate.
        camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
        camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)

        // We only need a spatial average, so pin zoom to the widest field and
        // disable HDR — both otherwise distort a clean PPG waveform. These are
        // always correct for PPG, so they aren't user-adjustable.
        camera.videoZoomFactor = 1.0
        if camera.activeFormat.isVideoHDRSupported {
            camera.automaticallyAdjustsVideoHDREnabled = false
            camera.isVideoHDREnabled = false
        }

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

        // Video stabilization warps frames over time and would inject motion
        // artifacts into the spatial average; disable it.
        if let connection = output.connection(with: .video),
           connection.isVideoStabilizationSupported {
            connection.preferredVideoStabilizationMode = .off
        }

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
        errorMessage = nil
        recordingState.startTime = nil
        recordingState.startPTS = nil
        captureSession = nil
        signalQuality = .noisy
        fingerDetected = false
        detectionStarted = false
        previousPPGValue = nil
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

                // Give auto-exposure / white balance a moment to settle on the
                // finger+torch scene, then lock them so they stop chasing (and
                // cancelling) the pulse. Runs on the serial session queue.
                self.sessionQueue.asyncAfter(deadline: .now() + 0.8) { [weak self] in
                    self?.lockExposureAndWhiteBalance()
                }

                Task { @MainActor in
                    self.captureSession = session
                    self.videoOutput = output
                    self.isRecording = true
                    // Recording runs indefinitely until the user stops it; there
                    // is no countdown to start.
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
                // User-selected level, clamped to what the hardware allows.
                // `setTorchModeOn(level:)` takes a Float, so compute in Float.
                let desiredLevel = min(max(Float(recordingState.tuning.torchLevel), 0.01),
                                       AVCaptureDevice.maxAvailableTorchLevel)
                try device.setTorchModeOn(level: desiredLevel)
            } else {
                device.torchMode = .off
            }
        } catch {
            throw CameraError.torchUnavailable(underlying: error)
        }
    }

    /// Freeze exposure and white balance after the sensor has settled on the
    /// finger+torch scene, so auto-exposure / AWB stop cancelling the pulse.
    nonisolated private func lockExposureAndWhiteBalance() {
        guard let device = Self.resolveCamera(id: recordingState.cameraID) else { return }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            if device.isExposureModeSupported(.locked) {
                device.exposureMode = .locked
            }
            if device.isWhiteBalanceModeSupported(.locked) {
                device.whiteBalanceMode = .locked
            }
        } catch {
            Task { @MainActor in
                self.errorMessage = "Failed to lock exposure: \(error.localizedDescription)"
            }
        }
    }

    /// Re-assert the torch if something (session reconfiguration, a transient
    /// interruption) has switched it back off while we are still recording.
    nonisolated private func ensureTorchOn() throws {
        guard let device = Self.resolveCamera(id: recordingState.cameraID),
              device.hasTorch, device.isTorchAvailable, !device.isTorchActive else { return }
        try setTorch(on: true)
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

        // Compute BPM from only the most recent window so the live readout tracks
        // the current heartbeat rather than averaging over the whole recording.
        let bpm = ppgProcessor.calculateBPM(
            from: samples,
            sampleRate: sampleRate,
            windowSeconds: liveBPMWindowSeconds
        )
        self.currentBPM = bpm

        // Refresh signal quality alongside BPM updates. `samples` is already a
        // bounded rolling window, so this stays recent and cheap.
        if !samples.isEmpty {
            self.signalQuality = ppgProcessor.assessQuality(samples: samples, sampleRate: sampleRate)
        }

        // Mark where the stable signal begins (once) so the saved measurement can
        // trim the noisy lead-in before BPM was first detected.
        if !detectionStarted, bpm != nil {
            markDetectionStart()
        }
    }

    /// Record the timestamp where a valid BPM was first detected so we can trim
    /// the noisy lead-in from the saved measurement.
    private func markDetectionStart() {
        detectionStarted = true
        self.recordingState.startTime = Date()
        self.detectionStartTimestamp = samples.last?.timestamp
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    /// Process each video frame to extract PPG signal. Frames that don't look
    /// like a fingertip over the torch (i.e. not mostly red/orange) are dropped
    /// and surfaced as a "place your finger" prompt instead of polluting the signal.
    nonisolated func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {

        // Extract per-frame colour stats (channel/ROI/filtering per tuning).
        guard let stats = extractFrameStats(from: sampleBuffer),
              Self.isFingerPresent(stats, minRedBlueRatio: recordingState.tuning.minRedBlueRatio) else {
            Task { @MainActor in self.setFingerDetected(false) }
            return
        }

        // Anchor t=0 to the first valid (finger-present) frame so the noisy
        // "no finger yet" lead-in doesn't shift the timebase.
        if recordingState.startPTS == nil {
            recordingState.startPTS = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        }
        guard let startPTS = recordingState.startPTS else { return }

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let relative = CMTimeSubtract(pts, startPTS)
        let timestamp = CMTimeGetSeconds(relative)

        Task { @MainActor in
            self.appendSample(value: stats.ppgValue, timestamp: timestamp)
        }
    }
    
    /// Append a smoothed PPG sample on the main actor to avoid cross-thread state mutation.
    @MainActor
    private func appendSample(value: Double, timestamp: Double) {
        if !fingerDetected { fingerDetected = true }

        let smoothed = smoothPPGValue(value)
        let limited = clampJump(smoothed)
        let sample = PPGSample(timestamp: timestamp, value: limited)
        samples.append(sample)

        // The recording now runs indefinitely, so keep only a trailing window to
        // bound memory and per-frame processing cost.
        let cutoff = timestamp - maxRetainedSeconds
        if let first = samples.first, first.timestamp < cutoff {
            samples.removeAll { $0.timestamp < cutoff }
        }

        // Refresh filtered copy for UI and quality estimate
        let sampleRate = estimatedSampleRate()
        filteredSamples = ppgProcessor.filteredForDisplay(
            samples: samples,
            sampleRate: sampleRate
        )
        updateBPMIfNeeded()
        maybeReassertTorch()
    }

    /// Update the finger-present state; when the finger is lifted, mark the
    /// signal noisy so a stale "good" reading doesn't linger behind the prompt.
    @MainActor
    private func setFingerDetected(_ detected: Bool) {
        if fingerDetected != detected { fingerDetected = detected }
        if !detected {
            signalQuality = .noisy
        }
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
        
        guard let previous = previousPPGValue else {
            previousPPGValue = clamped
            return clamped
        }
        
        // Heavier smoothing on sudden jumps (likely motion noise)
        let jump = abs(clamped - previous)
        let alpha: Double = jump > 60 ? 0.18 : 0.32
        let blended = previous * (1 - alpha) + clamped * alpha
        previousPPGValue = blended
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
    
    /// Average the frame's colour channels (over the optional centered ROI, with
    /// optional dark/saturated pixel filtering) into `FrameStats`. Channel, ROI,
    /// and filtering are all driven by the user-adjustable tuning snapshot.
    nonisolated private func extractFrameStats(from sampleBuffer: CMSampleBuffer) -> FrameStats? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }

        let tuning = recordingState.tuning

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)

        // BGRA byte order: B=0, G=1, R=2, A=3.
        let usingRed = tuning.channel == .red

        // Optionally restrict to the centered 50% region to avoid edge glare
        // and lens vignetting.
        let xStart = tuning.useROI ? width / 4 : 0
        let xEnd = tuning.useROI ? width * 3 / 4 : width
        let yStart = tuning.useROI ? height / 4 : 0
        let yEnd = tuning.useROI ? height * 3 / 4 : height

        var redSum: Int = 0
        var greenSum: Int = 0
        var blueSum: Int = 0
        var validPixelCount: Int = 0

        // Light subsampling reduces noise while keeping per-frame cost low.
        let step = 2  // every other pixel is enough at this resolution
        for y in stride(from: yStart, to: yEnd, by: step) {
            let rowStart = y * bytesPerRow
            for x in stride(from: xStart, to: xEnd, by: step) {
                let base = rowStart + x * 4
                let blue = Int(buffer[base])
                let green = Int(buffer[base + 1])
                let red = Int(buffer[base + 2])
                // Optionally drop under-exposed / saturated pixels (judged on the
                // selected PPG channel) that carry no pulse.
                let channelValue = usingRed ? red : green
                if tuning.filterPixels && (channelValue < 10 || channelValue > 245) { continue }
                redSum += red
                greenSum += green
                blueSum += blue
                validPixelCount += 1
            }
        }

        guard validPixelCount > 0 else { return nil }  // Need at least one usable pixel

        let count = Double(validPixelCount)
        let meanRed = Double(redSum) / count
        let meanGreen = Double(greenSum) / count
        return FrameStats(
            ppgValue: usingRed ? meanRed : meanGreen,
            meanRed: meanRed,
            meanGreen: meanGreen,
            meanBlue: Double(blueSum) / count
        )
    }

    /// Decide whether a fingertip is covering the lens. Light transmitted
    /// through a finger over the torch is overwhelmingly red/orange: the red
    /// channel is bright and clearly dominates blue (which tissue/blood absorb),
    /// and red is at least as strong as green. `minRedBlueRatio` sets how fully
    /// the finger must cover the lens — partial coverage lets in ambient light,
    /// raising blue and lowering the ratio. Anything else (lens pointed at a
    /// neutral or blue-ish scene, or too dark) is treated as "no finger".
    nonisolated private static func isFingerPresent(_ stats: FrameStats, minRedBlueRatio: Double) -> Bool {
        stats.meanRed > 40
            && stats.meanRed > stats.meanBlue * minRedBlueRatio
            && stats.meanRed >= stats.meanGreen
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
