LOGIN_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Login - Smart Attendance System</title>
    <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --border: rgba(148, 163, 184, 0.18);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.16), transparent 28%),
                var(--bg);
            color: var(--text);
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding: 20px;
        }
        .login-card {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.95), rgba(15, 23, 42, 0.95));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 40px;
            width: 100%;
            max-width: 420px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        h1 {
            font-size: 1.75rem;
            margin: 0 0 8px 0;
            text-align: center;
        }
        p.subtitle {
            color: var(--muted);
            text-align: center;
            font-size: 0.95rem;
            margin: 0 0 28px 0;
        }
        form { display: grid; gap: 18px; }
        label { display: grid; gap: 6px; color: var(--muted); font-size: 0.9rem; }
        input[type=text], input[type=password] {
            width: 100%;
            border: 1px solid var(--border);
            background: rgba(15, 23, 42, 0.8);
            color: var(--text);
            border-radius: 12px;
            padding: 12px 14px;
            outline: none;
            font-size: 1rem;
        }
        input[type=text]:focus, input[type=password]:focus {
            border-color: var(--accent);
        }
        .btn {
            border: 0;
            border-radius: 12px;
            padding: 14px;
            font-size: 1rem;
            font-weight: 700;
            color: #06111f;
            background: var(--accent);
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 8px;
        }
        .btn:hover { background: var(--accent-hover); }
        .flash {
            margin-bottom: 20px;
            padding: 12px 14px;
            border-radius: 12px;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #fca5a5;
            font-size: 0.9rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="login-card">
        <h1>Smart Attendance</h1>
        <p class="subtitle">Please log in to continue</p>

        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="flash">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <form method="post" action="{{ url_for('login') }}">
            <label>Username
                <input type="text" name="username" required placeholder="Enter username" autofocus>
            </label>
            <label>Password
                <input type="password" name="password" required placeholder="Enter password">
            </label>
            <button class="btn" type="submit">Sign In</button>
        </form>
    </div>
</body>
</html>
"""

STUDENT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Student Portal</title>
    <style>
        :root {
            --bg: #0f172a;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --border: rgba(148, 163, 184, 0.18);
            --accent: #38bdf8;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.16), transparent 28%),
                var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 32px;
            border-bottom: 1px solid var(--border);
            background: rgba(17, 24, 39, 0.8);
            backdrop-filter: blur(10px);
        }
        h1 { font-size: 1.25rem; margin: 0; }
        .user-info {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .badge {
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: var(--accent);
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .logout-btn {
            color: var(--muted);
            text-decoration: none;
            font-size: 0.9rem;
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid var(--border);
            transition: all 0.2s;
        }
        .logout-btn:hover {
            color: var(--text);
            background: rgba(255, 255, 255, 0.05);
        }
        main {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px;
        }
        .blank-card {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.8), rgba(15, 23, 42, 0.8));
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 60px;
            text-align: center;
            max-width: 500px;
            width: 100%;
        }
        .blank-card p {
            color: var(--muted);
            font-size: 1.1rem;
            margin: 0;
        }
    </style>
</head>
<body>
    <header>
        <h1>Smart Attendance System</h1>
        <div class="user-info">
            <span class="badge">Student Access</span>
            <a href="{{ url_for('logout') }}" class="logout-btn">Logout</a>
        </div>
    </header>
    <main>
        <div class="blank-card">
            <p>Welcome to Student Portal.</p>
        </div>
    </main>
</body>
</html>
"""
