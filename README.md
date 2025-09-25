# 🎓 AI-Powered Online Proctoring System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange)<img width="1024" height="1536" alt="modearc" src="https://github.com/user-attachments/assets/47135611-5ee6-42cd-a26e-c268dcd22b0d" />

![License](https://img.shields.io/badge/License-MIT-lightgrey)

> 🚀 An **AI-driven online exam proctoring system** that uses **computer vision, voice analysis, and real-time monitoring** to ensure **fair and secure remote assessments**.

---

## 📌 Features

✅ **Secure Authentication** – Candidate login/registration via Flask backend.  
✅ **Face Identification (20s verification)** – Verify candidate identity before exam.  
✅ **Head Pose Estimation** – Detect if candidate looks away from screen.  
✅ **Multiple Face Detection** – Ensure no additional persons are in the frame.  
✅ **Eye & Mouth Detection** – Identify suspicious movements (talking, reading).  
✅ **Object Detection (Phones/Notes)** – Detect cheating materials using YOLO.  
✅ **Voice Recognition** – Identify if candidate is speaking during exam.  
✅ **Event Logging & Screenshots** – Record suspicious events with timestamp + evidence.  
✅ **Real-time Event Panel** – Invigilators see alerts instantly in browser.  
✅ **Auto Submission** – Exam ends automatically if violations exceed threshold.  

---

## 🖥️ System Architecture

```mermaid
flowchart<img width="1024" height="1536" alt="modearc" src="https://github.com/user-attachments/assets/211913b2-e339-4ea5-99f2-ca8441d6482e" />
 LR
    A[Candidate Login] --> B[20s Face Identification]
    B --> C[Exam Interface + Webcam Feed]
    C --> D[Processing Module]
    D -->|Face Detection| E[Logs + Screenshots]
    D -->|Voice Recognition| E
    D -->|Object Detection| E
    D -->|Head Pose & Eye Tracking| E
    E --> F[Real-time Alerts to UI]
    E --> G[Auto-Submit Exam]



