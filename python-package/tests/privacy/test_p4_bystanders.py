"""P4 - Biometric processing of people who are not enrolled (flows 1 and 2).

Privacy issue: the door camera sees employees, visitors and passers-by.
``PersonAnalysis.get`` computes a face embedding and a body re-ID feature for
everybody in the frame, and ``match`` returns them inside the result even when
nobody matched (``person_id is None``). Any caller that logs or serialises the
result (``MatchResult.to_dict()``) therefore keeps biometric templates of people
who have no relationship with the system and never consented.

GDPR: Art. 5(1)(c) data minimisation, Art. 9 (no legal basis for bystanders),
Art. 25 (data protection by default).

The models are replaced by deterministic fakes, as in
``tests/person/test_person_analysis_minimal.py``, so no weights are needed.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from insightface.app import PersonAnalysis

FRAME = np.full((200, 260, 3), 80, np.uint8)
EMPLOYEE_FACE = np.array([1.0, 0.0, 0.0])
VISITOR_FACE = np.array([0.0, 1.0, 0.0])


@pytest.fixture
def door_camera(monkeypatch):
    state = SimpleNamespace(face_feature=EMPLOYEE_FACE, body_feature=np.array([0.0, 1.0, 0.0, 0.0]))

    class BodyDetector:
        def detect(self, image):
            return np.array([[0, 0, 100, 190, 0.95]], np.float32)

    class FaceDetector:
        static_input_size = None
        input_size = (640, 640)

        def detect(self, image, **kwargs):
            boxes = np.array([[20, 20, 70, 75, 0.99]], np.float32)
            points = np.array([[[30, 35], [55, 35], [43, 45], [34, 58], [54, 58]]], np.float32)
            return boxes, points

    class FaceRecognizer:
        output_shape = [1, 3]

        def get(self, image, face):
            return state.face_feature.copy()

    class BodyRecognizer:
        descriptor = {"outputs": [{"shape": [1, 4]}]}

        def embed(self, image):
            return state.body_feature.copy()

    def fake_models(name, root, providers, threads=4, body_det_size=0, face_det_size=0):
        return dict(detector=BodyDetector(), reid=BodyRecognizer(), face_detector=FaceDetector(),
                    face_recognizer=FaceRecognizer(), face_model_id="face-v1", reid_model_id="body-v1")

    monkeypatch.setattr("insightface.app.person_analysis.prepare_models", fake_models)
    app = PersonAnalysis(name="cheetah_s")
    accepted = app.register("employee_001", FRAME).accepted  # enrollment photo of the employee
    assert accepted == 1
    return state, app


def test_unmatched_visitor_keeps_no_biometric_template(door_camera):
    state, app = door_camera
    state.face_feature = VISITOR_FACE  # a visitor walks past the door

    [result] = app.match(app.get(FRAME))
    assert result.person_id is None  # correctly not recognised

    observation = result.observation
    retained = []
    if observation.face is not None and observation.face.embedding is not None:
        retained.append("face embedding")
    if observation.reid_feature is not None:
        retained.append("body re-ID feature")
    assert not retained, (
        f"P4 OPEN: the result for a non-enrolled person still carries their "
        f"{' and '.join(retained)}; it can be logged or exported without any legal basis."
    )


def test_serialised_result_of_visitor_contains_no_biometrics(door_camera):
    state, app = door_camera
    state.face_feature = VISITOR_FACE

    [result] = app.match(app.get(FRAME))
    serialised = json.dumps(result.to_dict())

    assert '"embedding": [' not in serialised and '"reid_feature": [' not in serialised, (
        "P4 OPEN: MatchResult.to_dict() of a non-enrolled person exports their "
        "biometric vectors; anything written to logs or reports will contain them."
    )


def test_enrolled_employee_is_still_recognised(door_camera):
    """Utility guard: minimisation must not break access control (should PASS)."""
    _, app = door_camera
    [result] = app.match(app.get(FRAME))
    assert result.person_id == "employee_001" and result.matched_by == "face"
