import io
import zipfile
import pytest


class TestInfoEndpoints:
    def test_health_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, test_client):
        response = test_client.get("/health")
        assert response.json()["status"] == "ok"

    def test_supported_entities_returns_all_five(self, test_client):
        response = test_client.get("/supported-entities")
        assert response.status_code == 200
        entities = set(response.json()["supported_entities"])
        assert entities == {"PERSON", "EMAIL", "PHONE", "ADDRESS", "ID"}

    def test_operators_contains_three_modes(self, test_client):
        response = test_client.get("/operators")
        assert response.status_code == 200
        operators = set(response.json()["operators"])
        assert operators == {"replace", "mask", "pseudonymize"}


class TestDeidentifyText:
    def test_valid_request_returns_200(self, test_client):
        response = test_client.post(
            "/deidentify-text",
            json={"text": "Иван Петров написал письмо", "mode": "replace"},
        )
        assert response.status_code == 200

    def test_response_has_required_fields(self, test_client):
        response = test_client.post(
            "/deidentify-text",
            json={"text": "Тестовый текст для обработки"},
        )
        data = response.json()
        for field in ("anonymized_text", "entities_count", "mode", "text_length"):
            assert field in data, f"Отсутствует поле: {field!r}"

    def test_empty_text_returns_400(self, test_client):
        response = test_client.post(
            "/deidentify-text",
            json={"text": "   "},
        )
        assert response.status_code == 400

    def test_empty_string_returns_400(self, test_client):
        response = test_client.post(
            "/deidentify-text",
            json={"text": ""},
        )
        assert response.status_code == 400

    def test_invalid_entities_filter_returns_400(self, test_client):
        response = test_client.post(
            "/deidentify-text",
            json={"text": "Привет", "entities": ["НЕСУЩЕСТВУЮЩИЙ_ТИП"]},
        )
        assert response.status_code == 400

    @pytest.mark.parametrize("mode", ["replace", "mask", "pseudonymize"])
    def test_all_modes_accepted(self, test_client, mode):
        response = test_client.post(
            "/deidentify-text",
            json={"text": "Тестовый текст", "mode": mode},
        )
        assert response.status_code == 200, f"Режим {mode!r} вернул ошибку"

    def test_entities_filter_accepted(self, test_client):
        response = test_client.post(
            "/deidentify-text",
            json={"text": "Привет", "entities": ["PERSON", "EMAIL"]},
        )
        assert response.status_code == 200

    def test_threshold_boundary_values(self, test_client):
        for threshold in (0.0, 0.5, 1.0):
            response = test_client.post(
                "/deidentify-text",
                json={"text": "Тест", "threshold": threshold},
            )
            assert response.status_code == 200, f"threshold={threshold} вернул ошибку"


class TestDeidentifyFile:
    def test_valid_utf8_file_returns_200(self, test_client):
        content = "Клиент Иван Петров, тел. +7 (999) 123-45-67".encode("utf-8")
        response = test_client.post(
            "/deidentify-file",
            files={"file": ("client.txt", content, "text/plain")},
            params={"mode": "replace"},
        )
        assert response.status_code == 200

    def test_file_too_large_returns_413(self, test_client):
        big_content = b"A" * (3 * 1024 * 1024)
        response = test_client.post(
            "/deidentify-file",
            files={"file": ("big.txt", big_content, "text/plain")},
        )
        assert response.status_code == 413

    def test_non_utf8_file_returns_400(self, test_client):
        bad_content = "Текст".encode("utf-16-le")
        response = test_client.post(
            "/deidentify-file",
            files={"file": ("bad_encoding.txt", bad_content, "text/plain")},
        )
        assert response.status_code == 400

    def test_utf8_bom_file_accepted(self, test_client):
        content = "\ufeffКлиент Иван Петров".encode("utf-8-sig")
        response = test_client.post(
            "/deidentify-file",
            files={"file": ("bom.txt", content, "text/plain")},
        )
        assert response.status_code == 200


class TestDeidentifyArchive:

    @staticmethod
    def _make_zip(files: dict[str, str]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, content in files.items():
                zf.writestr(name, content.encode("utf-8"))
        return buf.getvalue()

    def test_valid_zip_returns_200(self, test_client):
        zip_bytes = self._make_zip({
            "doc1.txt": "Клиент Иван Петров",
            "doc2.txt": "Телефон: +7 (999) 000-00-00",
        })
        response = test_client.post(
            "/deidentify-archive",
            files={"file": ("docs.zip", zip_bytes, "application/zip")},
        )
        assert response.status_code == 200

    def test_files_count_matches(self, test_client):
        zip_bytes = self._make_zip({
            "a.txt": "Текст первый",
            "b.txt": "Текст второй",
            "c.txt": "Текст третий",
        })
        response = test_client.post(
            "/deidentify-archive",
            files={"file": ("batch.zip", zip_bytes, "application/zip")},
        )
        assert response.json()["files_count"] == 3

    def test_non_zip_file_returns_400(self, test_client):
        response = test_client.post(
            "/deidentify-archive",
            files={"file": ("doc.txt", b"not a zip", "text/plain")},
        )
        assert response.status_code == 400

    def test_corrupted_zip_returns_400(self, test_client):
        response = test_client.post(
            "/deidentify-archive",
            files={"file": ("bad.zip", b"PK\x03\x04garbage", "application/zip")},
        )
        assert response.status_code == 400

    def test_empty_zip_returns_400(self, test_client):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w"):
            pass
        response = test_client.post(
            "/deidentify-archive",
            files={"file": ("empty.zip", buf.getvalue(), "application/zip")},
        )
        assert response.status_code == 400
