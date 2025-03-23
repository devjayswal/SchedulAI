from flask import Flask, render_template
import json
import os

app = Flask(__name__)

@app.route("/")
def index():
    output_dir = os.path.join(os.getcwd(), "output")
    with open(os.path.join(output_dir, "student_timetable.json"), "r") as f:
        student_timetable = json.load(f)
    with open(os.path.join(output_dir, "faculty_timetable.json"), "r") as f:
        faculty_timetable = json.load(f)
    with open(os.path.join(output_dir, "room_timetable.json"), "r") as f:
        room_timetable = json.load(f)
    
    return render_template(
        "index.html",
        student_timetable=student_timetable,
        faculty_timetable=faculty_timetable,
        room_timetable=room_timetable
    )

if __name__ == "__main__":
    app.run(debug=True)
