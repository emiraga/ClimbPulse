//
//  CameraPreviewView.swift
//  ClimbPulse
//
//  Lightweight SwiftUI wrapper around AVCaptureVideoPreviewLayer so users can
//  see which camera is active and how their finger covers it.
//

import SwiftUI
import AVFoundation

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession?
    
    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        view.videoPreviewLayer.connection?.videoOrientation = .portrait
        return view
    }
    
    func updateUIView(_ uiView: PreviewView, context: Context) {
        // Only (re)assign when the session actually changes. Reassigning the same
        // session on every SwiftUI re-render reconfigures the running session
        // repeatedly, which resets the camera torch (flash) back off.
        if uiView.videoPreviewLayer.session !== session {
            uiView.videoPreviewLayer.session = session
        }
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }
    
    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        // swiftlint:disable:next force_cast
        return layer as! AVCaptureVideoPreviewLayer
    }
}
