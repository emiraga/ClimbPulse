//
//  RecordingView.swift
//  ClimbPulse
//
//  Live recording screen showing PPG waveform and live BPM. Runs until stopped.
//

import AVFoundation
import SwiftUI

struct RecordingView: View {
    @ObservedObject var cameraManager: CameraManager
    let onComplete: (Measurement?) -> Void

    @Environment(\.dismiss) private var dismiss
    @Environment(\.colorScheme) private var colorScheme

    @State private var showTuning = false

    // Theme colors
    // JYU-inspired palette: deep blue + vivid orange
    private let primaryBlue = Color(red: 0.0, green: 0.34, blue: 0.65)  // #0056A5
    private let darkBlue = Color(red: 0.02, green: 0.16, blue: 0.32)  // #042948
    private let accentOrange = Color(red: 1.0, green: 0.51, blue: 0.0)  // #FF8200
    private let accentYellow = Color(red: 1.0, green: 0.72, blue: 0.11)  // #FFB81C

    private var backgroundGradient: LinearGradient {
        let colors: [Color]
        if colorScheme == .dark {
            colors = [darkBlue, primaryBlue.opacity(0.75)]
        } else {
            colors = [primaryBlue, primaryBlue.opacity(0.65), accentOrange.opacity(0.6)]
        }
        return LinearGradient(colors: colors, startPoint: .topLeading, endPoint: .bottomTrailing)
    }

    var body: some View {
        ZStack {
            backgroundGradient
                .ignoresSafeArea()

            VStack(spacing: 26) {
                // Top bar: live signal-tuning access, kept clear of the ring.
                HStack {
                    Spacer()
                    Button {
                        showTuning = true
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(.white)
                            .frame(width: 42, height: 42)
                            .background(Color.white.opacity(0.18))
                            .clipShape(Circle())
                            .overlay(Circle().stroke(Color.white.opacity(0.3), lineWidth: 1))
                    }
                    .accessibilityLabel("Signal tuning")
                }
                .padding(.horizontal, 20)
                .padding(.top, 36)  // clear the dynamic island / status bar

                VStack(spacing: 8) {
                    Text("Please keep still")
                        .font(.system(size: 20, weight: .semibold, design: .rounded))
                        .foregroundColor(.white)

                    Text("Gently cover the rear camera and flash with your fingertip.")
                        .font(.system(size: 15, weight: .medium, design: .rounded))
                        .foregroundColor(.white.opacity(0.8))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 28)
                }

                // Let the user try a different lens mid-measurement — the one
                // closest to the torch usually gives the strongest signal.
                // Tapping a segment restarts the capture on the new camera.
                if cameraManager.availableCameras.count > 1 {
                    CameraSegmentedSelector(
                        cameras: cameraManager.availableCameras,
                        selectedID: cameraManager.selectedCameraID,
                        onSelect: { cameraManager.switchCamera(to: $0) },
                        accent: accentOrange
                    )
                    .padding(.horizontal, 20)
                }

                // Live camera preview shown inside the BPM ring so users see the correct camera to cover.
                // The live PPG waveform is overlaid on the preview once a finger is detected.
                BPMPreviewRing(
                    session: cameraManager.captureSession,
                    bpm: cameraManager.currentBPM,
                    quality: cameraManager.signalQuality,
                    samples: cameraManager.filteredSamples,
                    fingerDetected: cameraManager.fingerDetected,
                    accent: accentOrange,
                    glow: accentYellow
                )

                // Signal-strength readout kept right under the preview so the
                // feedback sits next to what the user is adjusting.
                FingerPlacementIndicator(
                    sampleCount: cameraManager.samples.count,
                    signalQuality: cameraManager.signalQuality,
                    fingerDetected: cameraManager.fingerDetected
                )

                // Recording controls (measurement runs until the user stops it)
                VStack(spacing: 10) {
                    HStack(alignment: .center) {
                        // Cancel (left)
                        Button(action: cancelRecording) {
                            Image(systemName: "xmark")
                                .font(.system(size: 16, weight: .bold))
                                .foregroundColor(.white)
                                .frame(width: 42, height: 42)
                                .background(Color.red.opacity(0.28))
                                .clipShape(Circle())
                                .overlay(
                                    Circle().stroke(Color.white.opacity(0.35), lineWidth: 1)
                                )
                        }
                        .accessibilityLabel("Cancel and discard")

                        Spacer()

                        // Live "measuring" indicator in place of the old countdown
                        Label("Measuring", systemImage: "waveform.path.ecg")
                            .font(.system(size: 18, weight: .semibold, design: .rounded))
                            .foregroundColor(.white)
                            .symbolEffect(.pulse, options: .repeating)

                        Spacer()

                        // Stop/save (right)
                        Button(action: finishEarly) {
                            Image(systemName: "stop.fill")
                                .font(.system(size: 16, weight: .bold))
                                .foregroundColor(.white)
                                .frame(width: 42, height: 42)
                                .background(Color.green.opacity(0.35))
                                .clipShape(Circle())
                                .overlay(
                                    Circle().stroke(Color.white.opacity(0.35), lineWidth: 1)
                                )
                        }
                        .accessibilityLabel("Stop now and save")
                    }
                    .padding(.horizontal, 20)
                }

                // Instructions
                Text("Keep your finger steady on the rear camera + flash")
                    .font(.system(size: 15, weight: .medium, design: .rounded))
                    .foregroundColor(.white.opacity(0.8))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 28)
                    .padding(.vertical, 12)
            }
            .padding(.vertical, 20)
        }
        .sheet(isPresented: $showTuning) {
            CaptureTuningView(cameraManager: cameraManager)
        }
        .onAppear {
            startRecording()
        }
        .onChange(of: cameraManager.isRecording) { _, isRecording in
            if !isRecording {
                // recording stopped early or naturally; UI will dismiss via onComplete callback
            }
        }
    }

    private func startRecording() {
        cameraManager.onRecordingComplete = { measurement in
            onComplete(measurement)
        }
        cameraManager.startRecording()
    }

    private func finishEarly() {
        // Stop and let the normal completion flow show results
        cameraManager.stopRecording()
    }

    private func cancelRecording() {
        // Stop and discard this attempt
        cameraManager.stopRecording()
        dismiss()
        onComplete(nil)
    }
}

// MARK: - Supporting Views

/// Single-row lens picker for the recording screen. Each lens is a segment the
/// user can switch to with one tap (no menu), styled to sit on the recording
/// gradient. Uses the compact `shortName` so all lenses fit on one row.
struct CameraSegmentedSelector: View {
    let cameras: [CameraOption]
    let selectedID: String?
    let onSelect: (String?) -> Void
    let accent: Color

    var body: some View {
        HStack(spacing: 4) {
            ForEach(cameras) { camera in
                let isSelected = camera.id == selectedID
                Button {
                    onSelect(camera.id)
                } label: {
                    Text(camera.shortName)
                        .font(.system(size: 15, weight: .semibold, design: .rounded))
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                        .foregroundColor(isSelected ? .white : .white.opacity(0.75))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(isSelected ? accent : Color.clear)
                        )
                }
                .buttonStyle(.plain)
                .accessibilityLabel(camera.name)
                .accessibilityAddTraits(isSelected ? .isSelected : [])
            }
        }
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color.white.opacity(0.14))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
        )
    }
}

struct FingerPlacementIndicator: View {
    let sampleCount: Int
    let signalQuality: SignalQuality
    let fingerDetected: Bool

    private var signalStrength: SignalStrength {
        if !fingerDetected {
            return .noFinger
        } else if sampleCount < 10 {
            return .none
        } else if signalQuality == .good {
            return .good
        } else {
            return .weak
        }
    }

    enum SignalStrength {
        case noFinger, none, weak, good

        var color: Color {
            switch self {
            case .noFinger: return .red
            case .none: return .gray
            case .weak: return .yellow
            case .good: return Color(red: 1.0, green: 0.51, blue: 0.0)
            }
        }

        var text: String {
            switch self {
            case .noFinger: return "Place your finger on the camera lens"
            case .none: return "No signal"
            case .weak: return "Adjust finger for a clearer signal"
            case .good: return "Signal looks good"
            }
        }
    }

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(signalStrength.color)
                .frame(width: 8, height: 8)

            Text(signalStrength.text)
                .font(.system(size: 13, weight: .medium, design: .rounded))
                .foregroundColor(.white.opacity(0.6))
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.white.opacity(0.1))
        .clipShape(Capsule())
    }
}

struct BPMPreviewRing: View {
    let session: AVCaptureSession?
    let bpm: Int?
    let quality: SignalQuality
    let samples: [PPGSample]
    let fingerDetected: Bool
    let accent: Color
    let glow: Color

    private var qualityText: String {
        switch quality {
        case .good:
            return "Signal looks good"
        case .noisy:
            return "Adjust finger for clearer signal"
        }
    }

    private var bpmText: String {
        if let bpm {
            return "\(bpm)"
        } else {
            return "--"
        }
    }

    var body: some View {
        ZStack {
            // Live preview with a soft tint so users see exactly which camera/flash to cover
            CameraPreviewView(session: session)
                .overlay(
                    LinearGradient(
                        colors: [
                            Color.black.opacity(0.45),
                            Color.red.opacity(0.2),
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                // Overlay the live PPG waveform directly on the preview, but only
                // once a finger is covering the lens (otherwise it's just noise).
                .overlay(alignment: .bottom) {
                    if fingerDetected {
                        PPGWaveformView(samples: samples)
                            .frame(height: 90)
                            .padding(.horizontal, 30)
                            .padding(.bottom, 34)
                            .transition(.opacity)
                    }
                }
                .clipShape(Circle())

            // Ring around the live preview
            Circle()
                .stroke(
                    AngularGradient(
                        colors: [
                            accent,
                            glow,
                            accent,
                        ],
                        center: .center
                    ),
                    style: StrokeStyle(lineWidth: 14, lineCap: .round)
                )

            // BPM readout — nudged up so it clears the waveform overlay at the bottom
            VStack(spacing: 4) {
                Image(systemName: "heart.fill")
                    .font(.system(size: 28, weight: .semibold))
                    .foregroundColor(.white)
                    .symbolEffect(.pulse, options: .repeating, value: bpm != nil)

                Text(bpmText)
                    .font(.system(size: 58, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                    .contentTransition(.numericText())

                Text("bpm")
                    .font(.system(size: 18, weight: .semibold, design: .rounded))
                    .foregroundColor(.white.opacity(0.85))
            }
            .offset(y: fingerDetected ? -34 : 0)
            .animation(.easeInOut(duration: 0.25), value: fingerDetected)
        }
        .frame(width: 280, height: 280)
        .shadow(color: glow.opacity(0.35), radius: 18, y: 10)
    }
}

#Preview {
    RecordingView(cameraManager: CameraManager()) { _ in }
}
