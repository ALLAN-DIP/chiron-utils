"""Unit tests for `RandomProposerPlayer`."""

from diplomacy import Game
from gameplay_framework import GamePlay
from tornado import testing
from tornado.testing import AsyncTestCase
from typing_extensions import Final

from chiron_utils.bots import RandomProposerPlayer

SOA_TEST_PARAMS: Final = {
    "num_message_rounds": 3,
}


class TestBots(AsyncTestCase):
    """Tests for `RandomProposerPlayer` bot."""

    @testing.gen_test
    def test_play_simple(self):  # type: ignore[no-untyped-def]
        """Test sending a single message in a local game."""
        game = Game()
        soa_bot = RandomProposerPlayer("FRANCE", game)
        yield soa_bot.send_message("FRANCE", "A PAR - BUR")

    @testing.gen_test
    def test_play(self):  # type: ignore[no-untyped-def]
        """Test playing a local 3-phase game with all `RandomProposerPlayer` bots."""
        game = Game()

        game_play = GamePlay(
            game,
            [
                RandomProposerPlayer("AUSTRIA", game, **SOA_TEST_PARAMS),
                RandomProposerPlayer("ENGLAND", game, **SOA_TEST_PARAMS),
                RandomProposerPlayer("FRANCE", game, **SOA_TEST_PARAMS),
                RandomProposerPlayer("RUSSIA", game, **SOA_TEST_PARAMS),
                RandomProposerPlayer("GERMANY", game, **SOA_TEST_PARAMS),
                RandomProposerPlayer("ITALY", game, **SOA_TEST_PARAMS),
                RandomProposerPlayer("TURKEY", game, **SOA_TEST_PARAMS),
            ],
            3,
        )

        yield game_play.play()
        print("finish test_play")
