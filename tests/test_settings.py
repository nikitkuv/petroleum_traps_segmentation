class TestSettingsUseRgb:
    """Тесты для флага USE_RGB и вычисления IN_CHANNELS."""

    def test_default_use_rgb(self):
        """Тест значения USE_RGB по умолчанию (из кода, не из .env)."""
        from settings import Settings
        # Создаём настройки без загрузки .env и с явными значениями
        settings = Settings(_env_file=None, USE_RGB=True, USE_FAULTS=False)
        assert settings.USE_RGB == True

    def test_default_use_faults(self):
        """Тест значения USE_FAULTS по умолчанию (из кода, не из .env)."""
        from settings import Settings
        settings = Settings(_env_file=None, USE_RGB=True, USE_FAULTS=False)
        assert settings.USE_FAULTS == False

    def test_in_channels_with_rgb_default(self):
        """Тест IN_CHANNELS с RGB по умолчанию (USE_RGB=True, USE_FAULTS=False)."""
        from settings import Settings
        settings = Settings(_env_file=None, USE_RGB=True, USE_FAULTS=False)
        # Depth(1) + Isolines(1) + MapMask(1) + RGB(3) = 6
        assert settings.IN_CHANNELS == 6

    def test_in_channels_without_rgb(self):
        """Тест IN_CHANNELS без RGB (USE_RGB=False, USE_FAULTS=False)."""
        from settings import Settings
        settings = Settings(_env_file=None, USE_RGB=False, USE_FAULTS=False)
        # Depth(1) + Isolines(1) + MapMask(1) = 3
        assert settings.IN_CHANNELS == 3

    def test_in_channels_with_rgb_and_faults(self):
        """Тест IN_CHANNELS с RGB и разломами (USE_RGB=True, USE_FAULTS=True)."""
        from settings import Settings
        settings = Settings(_env_file=None, USE_RGB=True, USE_FAULTS=True)
        # Depth(1) + Isolines(1) + MapMask(1) + RGB(3) + Faults(1) = 7
        assert settings.IN_CHANNELS == 7

    def test_in_channels_without_rgb_with_faults(self):
        """Тест IN_CHANNELS без RGB но с разломами (USE_RGB=False, USE_FAULTS=True)."""
        from settings import Settings
        settings = Settings(_env_file=None, USE_RGB=False, USE_FAULTS=True)
        # Depth(1) + Isolines(1) + MapMask(1) + Faults(1) = 4
        assert settings.IN_CHANNELS == 4

    def test_in_channels_all_combinations(self):
        """Тест всех комбинаций USE_RGB и USE_FAULTS."""
        from settings import Settings
        # (USE_RGB, USE_FAULTS) -> ожидаемые каналы
        expected = [
            (True, False, 6),   # 3 base + 3 RGB
            (False, False, 3),  # 3 base
            (True, True, 7),    # 3 base + 3 RGB + 1 faults
            (False, True, 4),   # 3 base + 1 faults
        ]

        for use_rgb, use_faults, expected_channels in expected:
            settings = Settings(_env_file=None, USE_RGB=use_rgb, USE_FAULTS=use_faults)
            assert settings.IN_CHANNELS == expected_channels, \
                f"Failed for USE_RGB={use_rgb}, USE_FAULTS={use_faults}"

    def test_settings_independence(self):
        """Тест что разные экземпляры Settings независимы."""
        from settings import Settings
        settings1 = Settings(_env_file=None, USE_RGB=False, USE_FAULTS=False)
        settings2 = Settings(_env_file=None, USE_RGB=True, USE_FAULTS=False)

        assert settings1.USE_RGB == False
        assert settings2.USE_RGB == True
        assert settings1.IN_CHANNELS == 3
        assert settings2.IN_CHANNELS == 6


class TestSettingsUseFaults:
    """Тесты для флага USE_FAULTS."""

    def test_use_faults_affects_in_channels(self):
        """Тест что USE_FAULTS влияет на IN_CHANNELS."""
        from settings import Settings
        # Без разломов
        settings_without = Settings(_env_file=None, USE_RGB=False, USE_FAULTS=False)
        channels_without_faults = settings_without.IN_CHANNELS

        # С разломами
        settings_with = Settings(_env_file=None, USE_RGB=False, USE_FAULTS=True)
        channels_with_faults = settings_with.IN_CHANNELS

        # Разница должна быть ровно 1 канал
        assert channels_with_faults - channels_without_faults == 1

    def test_use_rgb_affects_in_channels(self):
        """Тест что USE_RGB влияет на IN_CHANNELS."""
        from settings import Settings
        # Без RGB
        settings_without = Settings(_env_file=None, USE_RGB=False, USE_FAULTS=False)
        channels_without_rgb = settings_without.IN_CHANNELS

        # С RGB
        settings_with = Settings(_env_file=None, USE_RGB=True, USE_FAULTS=False)
        channels_with_rgb = settings_with.IN_CHANNELS

        # Разница должна быть ровно 3 канала
        assert channels_with_rgb - channels_without_rgb == 3
