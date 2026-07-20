//
//  CaptureTuningView.swift
//  ClimbPulse
//
//  User-adjustable PPG capture options (channel, ROI, pixel filtering, torch
//  brightness). Bound directly to the CameraManager so changes apply live while
//  measuring. Options that should always be a fixed way for PPG (locked
//  exposure/white balance, 30 fps, no HDR/stabilisation, 1× zoom) are handled in
//  CameraManager and deliberately not exposed here.
//

import SwiftUI

struct CaptureTuningView: View {
    @ObservedObject var cameraManager: CameraManager
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Channel", selection: $cameraManager.ppgChannel) {
                        ForEach(PPGChannel.allCases) { channel in
                            Text(channel.displayName).tag(channel)
                        }
                    }
                    .pickerStyle(.segmented)
                } header: {
                    Text("Signal channel")
                } footer: {
                    Text("Red has the best SNR for a fingertip pressed over the torch. Green suits reflectance-style placement.")
                }

                Section {
                    Toggle("Center region only (ROI)", isOn: $cameraManager.useROI)
                    Toggle("Skip dark / saturated pixels", isOn: $cameraManager.filterPixels)
                } header: {
                    Text("Sampling")
                } footer: {
                    Text("ROI averages only the middle of the frame to avoid edge glare and vignetting. Pixel filtering drops values that are too dark or blown out to carry a pulse.")
                }

                Section {
                    Slider(
                        value: $cameraManager.torchLevel,
                        in: CameraManager.minUserTorchLevel...CameraManager.maxUserTorchLevel,
                        step: 0.05
                    )
                    HStack {
                        Text("Level")
                        Spacer()
                        Text(cameraManager.torchLevel, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    Text("Torch brightness")
                } footer: {
                    Text("Lower is cooler and avoids saturation; raise it only if the signal is weak.")
                }

                Section {
                    Slider(
                        value: $cameraManager.fingerCoverageStrictness,
                        in: CameraManager.minFingerStrictness...CameraManager.maxFingerStrictness,
                        step: 0.1
                    )
                    HStack {
                        Text("Strictness")
                        Spacer()
                        Text(cameraManager.fingerCoverageStrictness, format: .number.precision(.fractionLength(1)))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    Text("Finger coverage")
                } footer: {
                    Text("How fully your finger must cover the lens before measuring. Raise this if partial coverage is being accepted; lower it if a good placement isn't detected.")
                }
            }
            .navigationTitle("Signal Tuning")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

#Preview {
    CaptureTuningView(cameraManager: CameraManager())
}
