import { useState } from "react";

function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch("http://127.0.0.1:5000/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json();
      if (data.success) {
        alert("✅ " + data.message);

        // 🚀 Automatically start test after login
        const res = await fetch("http://127.0.0.1:5000/start-test");
        const testData = await res.json();

        alert("🧪 " + testData.message);
        // No need for user to do anything else → test.py opens camera
      } else {
        alert("❌ " + data.message);
      }
    } catch (err) {
      console.error("Error logging in:", err);
      alert("⚠️ Could not connect to backend");
    }
  };

  return (
    <div className="form-container">
      <h2>Login</h2>
      <form onSubmit={handleLogin}>
        <input
          type="text"
          placeholder="Enter Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Enter Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button type="submit">Login</button>
      </form>
    </div>
  );
}

export default Login;
