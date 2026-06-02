APP_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Smart Attendance Dashboard</title>
    <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --panel-2: #1f2937;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --accent-2: #22c55e;
            --danger: #ef4444;
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
        }
        .wrap { max-width: 1180px; margin: 0 auto; padding: 32px 18px 48px; }
        .hero {
            display: grid;
            gap: 16px;
            grid-template-columns: 1.6fr 1fr;
            align-items: stretch;
            margin-bottom: 18px;
        }
        .card {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.92), rgba(15, 23, 42, 0.92));
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.24);
        }
        h1, h2, h3, p { margin: 0; }
        h1 { font-size: 2rem; line-height: 1.1; margin-bottom: 10px; }
        .lead { color: var(--muted); max-width: 66ch; line-height: 1.6; }
        .stats {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }
        .stat {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px;
        }
        .stat .label { color: var(--muted); font-size: 0.86rem; margin-bottom: 8px; }
        .stat .value { font-size: 1.05rem; font-weight: 700; }
        .grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
            margin-top: 16px;
        }
        form { display: grid; gap: 12px; }
        label { display: grid; gap: 6px; color: var(--muted); font-size: 0.95rem; }
        input[type=text] {
            width: 100%;
            border: 1px solid var(--border);
            background: rgba(15, 23, 42, 0.8);
            color: var(--text);
            border-radius: 12px;
            padding: 12px 14px;
            outline: none;
        }
        input[type=text]:focus { border-color: rgba(56, 189, 248, 0.7); }
        .row { display: flex; gap: 10px; flex-wrap: wrap; }
        .btn {
            border: 0;
            border-radius: 12px;
            padding: 12px 16px;
            cursor: pointer;
            font-weight: 700;
            color: #06111f;
            background: var(--accent);
        }
        .btn.secondary { background: #cbd5e1; }
        .btn.success { background: var(--accent-2); }
        .btn.danger { background: var(--danger); color: white; }
        .muted { color: var(--muted); }
        .flash {
            margin: 0 0 14px;
            padding: 12px 14px;
            border-radius: 12px;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.24);
            color: var(--text);
        }
        .list {
            display: grid;
            gap: 10px;
            max-height: 420px;
            overflow: auto;
        }
        .list-item {
            border: 1px solid var(--border);
            background: rgba(15, 23, 42, 0.55);
            border-radius: 14px;
            padding: 12px 14px;
        }
        .list-item strong { display: block; margin-bottom: 6px; }
        .badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.82rem;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.2);
            color: var(--text);
        }
        .footer-note { margin-top: 16px; color: var(--muted); font-size: 0.9rem; }
        a { color: var(--accent); }
        .feed-wrap { margin-top: 12px; }
        .feed-wrap img { width: 100%; border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.12); display: block; }
        @media (max-width: 900px) {
            .hero, .grid { grid-template-columns: 1fr; }
            .stats { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <div class="hero">
            <section class="card">
                <h1>Smart Attendance Dashboard</h1>
                <p class="lead">Use the forms below to register a person and inspect a saved face image. Recognition runs continuously in the background and automatically marks attendance.</p>
                <div class="feed-wrap">
                    <img id="liveFeed" src="{{ url_for('camera_snapshot') }}" alt="Live camera">
                </div>
                <div id="attendanceBanner" class="flash" style="margin-top: 12px; display: none;"></div>
            </section>
            <section class="card">
                <div class="stats">
                    <div class="stat">
                        <div class="label">Recognition</div>
                        <div class="value">{{ 'Running' if recognition_running else 'Restarting' }}</div>
                    </div>
                    <div class="stat">
                        <div class="label">Registered users</div>
                        <div class="value" id="registeredCount">{{ registered_count }}</div>
                    </div>
                </div>
            </section>
        </div>

        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="flash">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <div class="grid">
            <section class="card">
                <h2>Register User</h2>
                <p class="muted" style="margin: 8px 0 16px;">Capture a face from the live RTSP feed and store the embedding plus cropped photo in the database.</p>
                <form method="post" action="{{ url_for('register_user_route') }}">
                    <label>Name
                        <input type="text" name="name" required value="{{ selected_name or '' }}">
                    </label>
                    <label>Kerberos ID
                        <input type="text" name="entry_no" required value="{{ selected_entry_no or suggested_entry_no }}">
                    </label>
                    <label style="display:flex; align-items:center; gap:10px; color: var(--text);">
                        <input type="checkbox" name="overwrite" value="1" checked>
                        Allow another face sample for an existing entry
                    </label>
                    <div class="row">
                        <button class="btn" type="submit">Start registration</button>
                    </div>
                </form>
            </section>

            <section class="card">
                <h2>Show Registered Face</h2>
                <p class="muted" style="margin: 8px 0 16px;">Find and preview the saved cropped face image for a Kerberos ID.</p>
                <form method="get" action="{{ url_for('show_user_route') }}">
                    <label>Kerberos ID
                        <input type="text" name="entry_no" required>
                    </label>
                    <div class="row">
                        <button class="btn secondary" type="submit">Show saved photo</button>
                    </div>
                </form>
                {% if selected_entry_no %}
                    <div style="margin-top: 16px;">
                        <span class="badge">Selected: {{ selected_entry_no }}</span>
                        {% if selected_name %}<span class="badge">{{ selected_name }}</span>{% endif %}
                        {% if face_exists %}
                            <div style="margin-top: 12px;">
                                <img
                                    src="{{ url_for('face_image', entry_no=selected_entry_no) }}?t={{ selected_entry_no }}"
                                    alt="Registered face for {{ selected_name or selected_entry_no }}"
                                    style="width: 100%; max-width: 320px; border-radius: 14px; border: 1px solid rgba(148, 163, 184, 0.12); display: block;"
                                >
                            </div>
                        {% else %}
                            <p style="margin-top: 10px; color: var(--muted);">No saved photo found.</p>
                        {% endif %}
                    </div>
                {% endif %}
            </section>
        </div>

        <div class="footer-note">Pipeline state is preserved in <a href="{{ url_for('index') }}">this dashboard</a>. Recognition is kept running automatically once the app is up.</div>
    </div>
    <script>
        (function () {
            const feed = document.getElementById('liveFeed');
            const feedUrl = {{ url_for('camera_snapshot')|tojson }};
            const statusUrl = {{ url_for('status')|tojson }};
            const registeredCount = document.getElementById('registeredCount');
            const attendanceBanner = document.getElementById('attendanceBanner');
            let attendanceBannerTimer = null;
            let lastAttendanceEventId = null;

            function refreshFeed() {
                if (!feed) return;
                feed.src = feedUrl + '?t=' + Date.now();
            }

            async function refreshStatus() {
                try {
                    const response = await fetch(statusUrl + '?t=' + Date.now(), { cache: 'no-store' });
                    if (!response.ok) return;
                    const data = await response.json();
                    if (registeredCount) {
                        registeredCount.textContent = data.registered_users;
                    }
                    if (attendanceBanner) {
                        const attendanceEventId = data.last_attendance_event_id ?? null;
                        const attendanceMessage = data.last_attendance_message || '';
                        if (attendanceMessage && attendanceEventId !== null && attendanceEventId !== lastAttendanceEventId) {
                            lastAttendanceEventId = attendanceEventId;
                            attendanceBanner.textContent = attendanceMessage;
                            attendanceBanner.style.display = 'block';
                            if (attendanceBannerTimer) {
                                clearTimeout(attendanceBannerTimer);
                            }
                            attendanceBannerTimer = setTimeout(() => {
                                attendanceBanner.textContent = '';
                                attendanceBanner.style.display = 'none';
                                attendanceBannerTimer = null;
                            }, 5000);
                        } else {
                            attendanceBanner.textContent = '';
                            attendanceBanner.style.display = 'none';
                        }
                    }
                } catch (error) {
                    // Keep the previous snapshot if the status request fails.
                }
            }

            setInterval(refreshFeed, 150);
            setInterval(refreshStatus, 1500);
            refreshStatus();
            feed.addEventListener('error', function () {
                setTimeout(refreshFeed, 1000);
            });
        }());
    </script>
</body>
</html>
"""