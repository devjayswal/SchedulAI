# lets create a function that will conver the json to class object
from models.timetable import Timetable
from models import Course, Faculty, Classroom, User
from typing import Dict, Any, List, Optional

import json
from bson import ObjectId
from datetime import datetime


async def jsonToTimeTableObject( json_data: Dict[str, Any]) -> Timetable:
    # Convert JSON to Timetable object
    timetable = Timetable(
        id=str(json_data.get("_id", "")),
        name=json_data.get("name", ""),
        description=json_data.get("description", ""),
        courses=[Course(**course) for course in json_data.get("courses", [])],
        faculties=[Faculty(**faculty) for faculty in json_data.get("faculties", [])],
        classrooms=[Classroom(**classroom) for classroom in json_data.get("classrooms", [])],
        users=[User(**user) for user in json_data.get("users", [])],
    )

    return timetable