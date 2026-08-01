from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_migration_graph_has_one_head() -> None:
    migrations_dir = Path(__file__).parents[1] / "migrations"
    config = Config(str(migrations_dir / "alembic.ini"))
    config.set_main_option("script_location", str(migrations_dir))

    script = ScriptDirectory.from_config(config)

    assert script.get_heads() == ["a7c1b9e2d4f5"]
