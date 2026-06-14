import sqlite3
import json
from datetime import datetime

_conn   = sqlite3.connect("database.db", check_same_thread=False)
_cursor = _conn.cursor()

_cursor.executescript("""
CREATE TABLE IF NOT EXISTS detections (
    id                            INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_number               INTEGER UNIQUE,
    image_path                    TEXT,
    original_image                TEXT,
    objects                       TEXT,
    objects_detailed              TEXT,
    person_count                  INTEGER DEFAULT 0,
    vehicle_count                 INTEGER DEFAULT 0,
    traffic_light_count           INTEGER DEFAULT 0,
    traffic_light_detected        BOOLEAN DEFAULT 0,
    traffic_light_color           TEXT,
    traffic_light_countdown       INTEGER,
    traffic_light_countdown_detected BOOLEAN DEFAULT 0,
    traffic_light_confidence      TEXT,
    detection_count               INTEGER,
    timestamp                     DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sequence    ON detections(sequence_number);
CREATE INDEX IF NOT EXISTS idx_timestamp   ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_tl_detected ON detections(traffic_light_detected);
""")
_conn.commit()


def _row_to_dict(row):
    if row is None:
        return None
    return {
        "id":                            row[0],
        "sequence_number":               row[1],
        "image_path":                    row[2],
        "original_image":                row[3],
        "objects":                       json.loads(row[4]) if row[4] else [],
        "objects_detailed":              json.loads(row[5]) if row[5] else [],
        "person_count":                  row[6]  or 0,
        "vehicle_count":                 row[7]  or 0,
        "traffic_light_count":           row[8]  or 0,
        "traffic_light_detected":        bool(row[9]),
        "traffic_light_color":           row[10] or "unknown",
        "traffic_light_countdown":       row[11],
        "traffic_light_countdown_detected": bool(row[12]),
        "traffic_light_confidence":      row[13] or "low",
        "detection_count":               row[14] or 0,
        "timestamp":                     row[15],
    }


def save_detection(sequence_number, image_path, original_image, result):
    tl  = result['traffic_light']
    obj = json.dumps(result.get('objects', []))
    det = json.dumps(result.get('objects_detailed', []))

    _cursor.execute(
        "SELECT id FROM detections WHERE sequence_number = ?",
        (sequence_number,))
    if _cursor.fetchone():
        _cursor.execute("""
            UPDATE detections SET
                image_path=?, original_image=?, objects=?, objects_detailed=?,
                person_count=?, vehicle_count=?, traffic_light_count=?,
                traffic_light_detected=?, traffic_light_color=?,
                traffic_light_countdown=?, traffic_light_countdown_detected=?,
                traffic_light_confidence=?, detection_count=?
            WHERE sequence_number=?
        """, (
            image_path, original_image, obj, det,
            result['counts']['people'],
            result['counts']['vehicles'],
            result['counts']['traffic_lights'],
            tl['detected'], tl['color'], tl['countdown'],
            tl.get('countdown_detected', False),
            tl.get('confidence', 'low'),
            result['counts']['total'],
            sequence_number,
        ))
    else:
        _cursor.execute("""
            INSERT INTO detections
                (sequence_number, image_path, original_image, objects,
                 objects_detailed, person_count, vehicle_count,
                 traffic_light_count, traffic_light_detected,
                 traffic_light_color, traffic_light_countdown,
                 traffic_light_countdown_detected, traffic_light_confidence,
                 detection_count)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            sequence_number, image_path, original_image, obj, det,
            result['counts']['people'],
            result['counts']['vehicles'],
            result['counts']['traffic_lights'],
            tl['detected'], tl['color'], tl['countdown'],
            tl.get('countdown_detected', False),
            tl.get('confidence', 'low'),
            result['counts']['total'],
        ))
    _conn.commit()
    return _cursor.lastrowid


def get_latest():
    _cursor.execute(
        "SELECT * FROM detections ORDER BY sequence_number DESC LIMIT 1")
    return _row_to_dict(_cursor.fetchone())


def get_history(limit=50):
    _cursor.execute(
        "SELECT * FROM detections ORDER BY sequence_number DESC LIMIT ?",
        (limit,))
    return [_row_to_dict(r) for r in _cursor.fetchall()]


def get_detection_by_sequence(seq_num):
    _cursor.execute(
        "SELECT * FROM detections WHERE sequence_number=?", (seq_num,))
    return _row_to_dict(_cursor.fetchone())


def get_statistics():
    _cursor.execute("""
        SELECT
            COUNT(*)                                                AS total_detections,
            COALESCE(SUM(person_count), 0)                         AS total_people,
            COALESCE(SUM(vehicle_count), 0)                        AS total_vehicles,
            COALESCE(SUM(traffic_light_count), 0)                  AS total_tl,
            SUM(CASE WHEN traffic_light_detected=1 THEN 1 ELSE 0 END) AS detections_with_tl,
            SUM(CASE WHEN traffic_light_color='red'    THEN 1 ELSE 0 END) AS red_lights,
            SUM(CASE WHEN traffic_light_color='yellow' THEN 1 ELSE 0 END) AS yellow_lights,
            SUM(CASE WHEN traffic_light_color='green'  THEN 1 ELSE 0 END) AS green_lights,
            SUM(CASE WHEN traffic_light_countdown_detected=1 THEN 1 ELSE 0 END) AS countdown_hits,
            COALESCE(AVG(detection_count), 0)                      AS avg_objects
        FROM detections
    """)
    s = _cursor.fetchone()
    return {
        "total_detections":             s[0],
        "total_people":                 s[1],
        "total_vehicles":               s[2],
        "total_traffic_lights":         s[3],
        "detections_with_traffic_light": s[4],
        "red_lights":                   s[5],
        "yellow_lights":                s[6],
        "green_lights":                 s[7],
        "countdown_detected":           s[8],
        "avg_objects_per_detection":    round(s[9], 2),
    }


def reset_database():
    _cursor.executescript("""
        DROP TABLE IF EXISTS detections;
        CREATE TABLE detections (
            id                            INTEGER PRIMARY KEY AUTOINCREMENT,
            sequence_number               INTEGER UNIQUE,
            image_path                    TEXT,
            original_image                TEXT,
            objects                       TEXT,
            objects_detailed              TEXT,
            person_count                  INTEGER DEFAULT 0,
            vehicle_count                 INTEGER DEFAULT 0,
            traffic_light_count           INTEGER DEFAULT 0,
            traffic_light_detected        BOOLEAN DEFAULT 0,
            traffic_light_color           TEXT,
            traffic_light_countdown       INTEGER,
            traffic_light_countdown_detected BOOLEAN DEFAULT 0,
            traffic_light_confidence      TEXT,
            detection_count               INTEGER,
            timestamp                     DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX idx_sequence    ON detections(sequence_number);
        CREATE INDEX idx_timestamp   ON detections(timestamp);
        CREATE INDEX idx_tl_detected ON detections(traffic_light_detected);
    """)
    _conn.commit()
    print("Database reset successfully")