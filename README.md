Dual-Threshold Face Recognition and Secure Access System
Overview
This repository contains a high-security facial recognition system that implements a dual-confidence threshold model. Unlike basic detection scripts, this system provides tiered access control, allowing for "identity verification" at standard ranges and "secure access" only upon near-perfect matches.

Key Features
Dual-Threshold Logic: Implements two separate security tiers based on recognition distance scores.

Interactive Live Learning: Dynamically add new users and passwords during the live camera stream.

Persistent Data Management: User credentials and names are stored in a local database to ensure data survival across sessions.

Automated Security Termination: The application immediately closes upon a high-security match to protect displayed credentials.

Real-Time Identification: Provides live feedback on recognition distance and match status.

Technologies Used
Python: Core application logic.

OpenCV: Image processing and LBPH (Local Binary Patterns Histograms) face recognition.

Haar Cascade: High-speed frontal face detection.

NumPy & Pillow: Array manipulation and image format conversion.

Playsound: Audible feedback for security events.

System Architecture
The system operates through a specialized pipeline:

Detection: Captures frames and uses a Haar Cascade classifier to locate faces.

Analysis: Normalizes facial data into grayscale for the LBPH algorithm.

Classification: Compares data against the trained trainer.yml model.

Logic Execution: Applies threshold-based conditions to determine access levels.

Security Model
