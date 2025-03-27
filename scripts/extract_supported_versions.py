def extract_supported_versions(extensions):
    if isinstance(extensions, list):  # Проверяем, что это список
        for ext in extensions:
            if ext.get("name") == "supported_versions (43)":
                return ext.get("versions", [])  # Берем список версий
    return None  # Если не найдено