"""A collection of various utilities useful for building bots."""

import asyncio
import json
import logging
import os
from typing import Any, List, Mapping, Optional, Set

from daidepp import AnyDAIDEToken, DAIDEGrammar, create_daide_grammar, daide_visitor
from daidepp.grammar.grammar import MAX_DAIDE_LEVEL
from diplomacy import Game


def return_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Returns a properly set up logger.

    Args:
        name: Name of logger.
        log_level: Verbosity level of logger.

    Returns:
        An initialized logger.
    """
    # Monkey patch module to show milliseconds
    logging.Formatter.default_msec_format = "%s.%03d"

    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        # Fall back to `WARNING`, the default level of the root logger, if `log_level` is `NOTSET`
        level=log_level or logging.WARNING,
    )
    new_logger = logging.getLogger(name)
    new_logger.setLevel(log_level)
    return new_logger


logger = return_logger(__name__)

POWER_NAMES_DICT = {
    "AUS": "AUSTRIA",
    "ENG": "ENGLAND",
    "FRA": "FRANCE",
    "GER": "GERMANY",
    "ITA": "ITALY",
    "RUS": "RUSSIA",
    "TUR": "TURKEY",
}

mapping = {
    "adr": ["Adriatic Sea"],
    "aeg": ["Aegean Sea"],
    "alb": ["Albania"],
    "ank": ["Ankara"],
    "apu": ["Apulia"],
    "arm": ["Armenia"],
    "bal": ["Baltic Sea"],
    "bar": ["Barents Sea"],
    "bel": ["Belgium"],
    "ber": ["Berlin"],
    "bla": ["Black Sea"],
    "boh": ["Bohemia"],
    "bre": ["Brest"],
    "bud": ["Budapest"],
    "bul/ec": ["Bulgaria (East Coast)"],
    "bul/sc": ["Bulgaria (South Coast)"],
    "bul": ["Bulgaria"],
    "bur": ["Burgundy"],
    "cly": ["Clyde"],
    "con": ["Constantinople"],
    "den": ["Denmark"],
    "eas": ["Eastern Mediterranean"],
    "edi": ["Edinburgh"],
    "eng": ["English Channel"],
    "fin": ["Finland"],
    "gal": ["Galicia"],
    "gas": ["Gascony"],
    "gre": ["Greece"],
    "lyo": ["Gulf of Lyon"],
    "bot": ["Gulf of Bothnia"],
    "hel": ["Heligoland Bight"],
    "hol": ["Holland"],
    "ion": ["Ionian Sea"],
    "iri": ["Irish Sea"],
    "kie": ["Kiel"],
    "lvp": ["Liverpool"],
    "lvn": ["Livonia"],
    "lon": ["London"],
    "mar": ["Marseilles"],
    "mao": ["Mid-Atlantic Ocean"],
    "mos": ["Moscow"],
    "mun": ["Munich"],
    "nap": ["Naples"],
    "nao": ["North Atlantic Ocean"],
    "naf": ["North Africa"],
    "nth": ["North Sea"],
    "nwy": ["Norway"],
    "nwg": ["Norwegian Sea"],
    "par": ["Paris"],
    "pic": ["Picardy"],
    "pie": ["Piedmont"],
    "por": ["Portugal"],
    "pru": ["Prussia"],
    "rom": ["Rome"],
    "ruh": ["Ruhr"],
    "rum": ["Romania"],
    "ser": ["Serbia"],
    "sev": ["Sevastopol"],
    "sil": ["Silesia"],
    "ska": ["Skagerrak"],
    "smy": ["Smyrna"],
    "spa/nc": ["Spain (North Coast)"],
    "spa/sc": ["Spain (South Coast)"],
    "spa": ["Spain"],
    "stp/nc": ["St. Petersburg (North Coast)"],
    "stp/sc": ["St. Petersburg (South Coast)"],
    "stp": ["St. Petersburg"],
    "swe": ["Sweden"],
    "syr": ["Syria"],
    "tri": ["Trieste"],
    "tun": ["Tunis"],
    "tus": ["Tuscany"],
    "tyr": ["Tyrolia"],
    "tys": ["Tyrrhenian Sea"],
    "ukr": ["Ukraine"],
    "ven": ["Venice"],
    "vie": ["Vienna"],
    "wal": ["Wales"],
    "war": ["Warsaw"],
    "wes": ["Western Mediterranean"],
    "yor": ["Yorkshire"],
    "s":["Support"],
    "a":["Army at"],
    "f":["Fleet at"],
    "[a":["[Army at"]
}


# Option for debugging without specialized builds
DEBUG_MODE = False
if os.environ.get("ALLAN_DEBUG") is not None:
    logger.info("Enabling debugging mode")
    DEBUG_MODE = True

MESSAGE_GRAMMAR = create_daide_grammar(level=MAX_DAIDE_LEVEL, string_type="message")
ALL_GRAMMAR = create_daide_grammar(level=MAX_DAIDE_LEVEL, string_type="all")


def is_valid_daide_message(string: str, grammar: Optional[DAIDEGrammar] = None) -> bool:
    """Determines whether a string is a valid DAIDE message.

    Args:
        string: String to check for valid DAIDE.
        grammar: DAIDE grammar to use. Defaults to complete message grammar.

    Returns:
        Whether the string is valid DAIDE or not.
    """
    if grammar is None:
        grammar = MESSAGE_GRAMMAR
    try:
        parse_tree = grammar.parse(string)
        daide_visitor.visit(parse_tree)
    except asyncio.CancelledError:  # pylint: disable=try-except-raise
        raise
    except Exception:  # noqa: BLE001
        return False
    return True


def parse_daide(string: str) -> AnyDAIDEToken:
    """Parses a DAIDE string into `daidepp` objects.

    Args:
        string: String to parse into DAIDE.

    Returns:
        Parsed DAIDE object.

    Raises:
        ValueError: If string is invalid DAIDE.
    """
    try:
        parse_tree = ALL_GRAMMAR.parse(string)
        return daide_visitor.visit(parse_tree)
    except asyncio.CancelledError:  # pylint: disable=try-except-raise
        raise
    except Exception as ex:
        raise ValueError(f"Failed to parse DAIDE string: {string!r}") from ex


def serialize_message_dict(message_dict: Mapping[str, Any]) -> str:
    """Serialize suggestion message in a consistent format."""
    # `separators` is used here to write JSON compactly
    return json.dumps(message_dict, ensure_ascii=False, separators=(",", ":"))


def get_order_tokens(order: str) -> List[str]:
    """Retrieves the order tokens used in an order.

    e.g., "A PAR - MAR" would return ["A PAR", "-", "MAR"]
    NOTE: Stolen from `diplomacy_research` repository

    Args:
        order: DipNet order.

    Returns:
        List of tokens from DipNet order.
    """
    # We need to keep 'A', 'F', and '-' in a temporary buffer to concatenate them with the next word
    # We replace 'R' orders with '-'
    # Tokenization would be: 'A PAR S A MAR - BUR' --> 'A PAR', 'S', 'A MAR', '- BUR'
    #                        'A PAR R MAR'         --> 'A PAR', '- MAR'
    buffer, order_tokens = [], []
    for word in order.replace(" R ", " - ").split():
        buffer += [word]
        if word not in {"A", "F", "-"}:
            order_tokens.append(" ".join(buffer))
            buffer = []
    return order_tokens


def get_other_powers(powers: List[str], game: Game) -> Set[str]:
    """Get all powers not provided in input.

    Args:
        powers: Powers to not return.
        game: Game to retrieve powers from.

    Returns:
        Powers in the game other than those given.
    """
    return set(game.get_map_power_names()) - set(powers)


def neighboring_opps(
    game: Game,
    power_name: str,
    opponents: List[str],
) -> List[str]:
    """Retrieve a list of neighboring opponents of a given power.

    Args:
        game: Game to retrieve information from.
        power_name: Power to find neighboring opponents of.
        opponents: List of opponents of given power.

    Returns:
        List of opponents that are neighbors of the given power.
    """
    neighbors = set()  # set to prevent duplicates
    adj_provs = set()  # provs adjacent to power_name territories
    for prov in game.powers[power_name].influence:
        adj_provs.update({x.upper() for x in game.map.abut_list(prov)})

    for opponent in opponents:
        for prov in game.powers[opponent].influence:
            if prov in adj_provs:
                neighbors.add(opponent)
                break
    return sorted(neighbors)
