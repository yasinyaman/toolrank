import random

from toolrank.names import API_NAME_RE, api_name


def test_readable_ids_keep_their_shape():
    assert api_name("github/issues/create") == "github__issues__create"
    assert api_name("time/get_current_time") == "time__get_current_time"
    assert api_name("everything/get-sum") == "everything__get-sum"
    assert api_name("stripe/PostRefunds") == "stripe__PostRefunds"


def test_other_ids_are_hashed_and_still_distinct():
    spaced = api_name("swagger-petstore/find pet by id")
    assert spaced.startswith("swagger-petstore__find_pet_by_id___") and API_NAME_RE.match(spaced)
    long = api_name("github/" + "x" * 80)
    assert len(long) == 64 and "___" in long and API_NAME_RE.match(long)
    # these would all flatten to the same string: underscores next to a slash, or doubled
    tricky = ["a_/b", "a/_b", "a__b", "a/b", "a b", "a_b", ".", ""]
    names = [api_name(t) for t in tricky]
    assert len(set(names)) == len(tricky) and names[3] == "a__b" and names[5] == "a_b"
    assert all(API_NAME_RE.match(n) for n in names)


def test_names_are_safe_injective_and_stable_on_random_ids():
    rng = random.Random(7)
    alphabet = "ab-_/ .é:"
    ids = {"".join(rng.choice(alphabet) for _ in range(rng.randint(0, 90))) for _ in range(5000)}
    names = {t: api_name(t) for t in ids}
    assert all(API_NAME_RE.match(n) for n in names.values())
    assert len(set(names.values())) == len(ids)
    assert all(api_name(t) == n for t, n in names.items())
