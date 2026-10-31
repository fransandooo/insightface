"""P1 - Plaintext, linkable biometric templates (flows 4 and 5, zone Z4).

Privacy issue: ``Storage.add_face_sample`` stores the ArcFace embedding with
``embedding_to_blob`` as raw float32 bytes. Anyone who can read the database
file (curious administrator, stolen backup) obtains the employee's biometric
template. Because the recognition model is public, the stolen vector can be
compared with embeddings of photos found online (linkability) or inverted into
an approximate face image.

GDPR: Art. 9 (special category data), Art. 32 (security of processing),
Art. 25 (data protection by design).
"""
import sqlite3

import numpy as np

FACE_SIMILARITY_THRESHOLD = 0.45  # PersonConfig.face_similarity_threshold


def test_stored_template_is_not_readable_in_plaintext(gallery, employee_embedding, raw_bytes):
    storage, db_path = gallery
    person_id = storage.add_person("Alice")
    storage.add_face_sample(person_id, employee_embedding, model_name="cheetah_s")

    assert employee_embedding.tobytes() not in raw_bytes(db_path), (
        "P1 OPEN: the employee's face embedding is stored in plaintext in "
        "insightface_gui.db; anyone with file access to zone Z4 can copy it."
    )


def test_stolen_template_cannot_be_linked_with_the_public_model(gallery, employee_embedding):
    storage, db_path = gallery
    person_id = storage.add_person("Alice")
    storage.add_face_sample(person_id, employee_embedding, model_name="cheetah_s")

    # The adversary reads the BLOB directly and interprets it as float32,
    # then compares it with the embedding the public ArcFace model produces
    # for a photo of Alice found online (here: the same vector).
    with sqlite3.connect(db_path) as conn:
        blob = conn.execute("SELECT embedding FROM face_samples").fetchone()[0]
    usable = len(blob) - len(blob) % 4
    stolen = np.frombuffer(blob[:usable], dtype=np.float32)[: employee_embedding.size]
    if stolen.size < employee_embedding.size or not np.isfinite(stolen).all():
        return  # not interpretable as a template: unlinkable
    similarity = float(np.dot(stolen, employee_embedding) / (np.linalg.norm(stolen) + 1e-12))

    assert similarity < FACE_SIMILARITY_THRESHOLD, (
        f"P1 OPEN: a template stolen from the database matches the public-model "
        f"embedding of the same person (cosine {similarity:.2f} >= "
        f"{FACE_SIMILARITY_THRESHOLD}); employees can be re-identified outside "
        f"the workplace."
    )


def test_protected_templates_still_identify_the_employee(gallery, employee_embedding):
    """Utility guard: a fix for P1 must not break recognition (should PASS)."""
    storage, _ = gallery
    person_id = storage.add_person("Alice")
    storage.add_face_sample(person_id, employee_embedding, model_name="cheetah_s")

    results = storage.search_embeddings(
        employee_embedding, top_k=1, threshold=FACE_SIMILARITY_THRESHOLD, model_name="cheetah_s"
    )
    assert results and results[0].person_id == person_id
