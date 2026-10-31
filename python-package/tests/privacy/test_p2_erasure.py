"""P2 - Incomplete erasure of a former employee's biometric data (zone Z4).

Privacy issue: the People Library deletes a person with
``storage.delete_person(person_id, delete_samples=False)``
(``insightface/gui/pages/people_library_page.py``). That only sets
``face_samples.person_id`` to NULL: the embedding, landmarks and crop path stay
in the database and the face crop stays on disk. Even with
``delete_samples=True`` SQLite keeps the deleted bytes in free pages unless
``secure_delete`` or ``VACUUM`` is used, so the template remains recoverable
from the file or its backups.

GDPR: Art. 17 (right to erasure), Art. 5(1)(e) (storage limitation).
"""
import ast
from pathlib import Path

import insightface

PEOPLE_LIBRARY_PAGE = Path(insightface.__file__).parent / "gui" / "pages" / "people_library_page.py"


def _enroll(storage, embedding, crop_path):
    person_id = storage.add_person("Alice")
    sample_id = storage.add_face_sample(
        person_id, embedding, crop_path=str(crop_path), source_image_path=str(crop_path),
        model_name="cheetah_s",
    )
    return person_id, sample_id


def test_deleting_an_employee_removes_their_face_samples(gallery, employee_embedding, face_crop, query_one):
    storage, db_path = gallery
    person_id, sample_id = _enroll(storage, employee_embedding, face_crop)

    storage.delete_person(person_id)  # default call, as used by the application

    remaining = query_one(db_path, "SELECT COUNT(*) FROM face_samples WHERE id = ?", (sample_id,))
    assert remaining == 0, (
        "P2 OPEN: after deleting the employee, their face sample (embedding, "
        "landmarks, crop path) is still stored as an orphan row."
    )


def test_deleting_an_employee_removes_their_face_crop_file(gallery, employee_embedding, face_crop):
    storage, _ = gallery
    person_id, _ = _enroll(storage, employee_embedding, face_crop)

    storage.delete_person(person_id, delete_samples=True)  # best case available today

    assert not face_crop.exists(), (
        "P2 OPEN: the face crop image of a deleted employee remains in the crops/ folder."
    )


def test_deleted_template_cannot_be_recovered_from_the_database_file(
    gallery, employee_embedding, face_crop, raw_bytes
):
    storage, db_path = gallery
    person_id, _ = _enroll(storage, employee_embedding, face_crop)

    storage.delete_person(person_id, delete_samples=True)  # best case available today

    assert employee_embedding.tobytes() not in raw_bytes(db_path), (
        "P2 OPEN: the deleted template is still present in the SQLite file's "
        "free pages (no secure_delete / VACUUM); forensic recovery is possible."
    )


def test_people_library_requests_full_deletion():
    """The GUI must not ask the storage layer to keep the samples."""
    source = PEOPLE_LIBRARY_PAGE
    tree = ast.parse(source.read_text(encoding="utf-8"))
    offending = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "delete_person"
        and any(
            kw.arg == "delete_samples" and isinstance(kw.value, ast.Constant) and kw.value.value is False
            for kw in node.keywords
        )
    ]
    assert not offending, (
        f"P2 OPEN: {source.name} line(s) {offending} delete a person with "
        f"delete_samples=False, leaving their biometric samples behind."
    )
