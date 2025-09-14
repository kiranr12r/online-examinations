import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

export default function Exam() {
  const location = useLocation();
  const username = location.state?.username || "candidate";
  const [events, setEvents] = useState([]);

  useEffect(() => {
    const es = new EventSource("http://127.0.0.1:5000/events");
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        // filter events for this candidate
        if (!data.candidate || data.candidate === username) {
          setEvents(prev => [data, ...prev].slice(0, 30)); // keep last 30
        }
      } catch (err) {
        // ignore parse errors and keepalive comments
      }
    };
    es.onerror = (err) => {
      console.error("EventSource error:", err);
      // es.close(); // optionally reconnect logic
    };
    return () => {
      es.close();
    };
  }, [username]);

  return (
    <div style={{ padding: 20 }}>
      <div style={{ display: "flex", gap: 20 }}>
        <div style={{ flex: 2 }}>
          <h2>Exam Monitoring — {username}</h2>
          <img
            src={`http://127.0.0.1:5000/video_feed?candidate=${encodeURIComponent(username)}`}
            alt="Live Camera"
            style={{ width: "100%", maxHeight: "640px", borderRadius: 10, border: "2px solid #1e3a8a" }}
          />
        </div>

        <div style={{ flex: 1 }}>
          <div style={{ background: "#fff", padding: 12, borderRadius: 8, border: "1px solid #ddd", height: 640, overflowY: "auto" }}>
            <h3 style={{ color: "#1e3a8a" }}>Suspicious Events</h3>
            {events.length === 0 ? (
              <p className="no-events">No suspicious activity detected ✅</p>
            ) : (
              events.map((ev, idx) => (
                <div key={idx} className="event-card">
                  <div><b>{ev.time}</b></div>
                  <div>{ev.event}</div>
                  {ev.details ? <div style={{ fontSize: 13, color: "#333" }}>{ev.details}</div> : null}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
