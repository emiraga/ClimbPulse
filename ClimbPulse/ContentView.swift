//
//  ContentView.swift
//  ClimbPulse
//
//  Main content view - entry point for the app UI.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var measurementStore = MeasurementStore()

    var body: some View {
        HomeView()
            .environmentObject(measurementStore)
    }
}

#Preview {
    ContentView()
}
