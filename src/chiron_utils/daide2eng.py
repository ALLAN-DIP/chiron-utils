"""Utilities for converting from DAIDE syntax to English."""

from collections.abc import Sequence
from typing import Union, cast

from daidepp import (
    ALYVSS,
    AND,
    ANG,
    BCC,
    BLD,
    BWX,
    CCL,
    CHO,
    CVY,
    DMZ,
    DRW,
    DSB,
    EXP,
    FCT,
    FOR,
    FRM,
    FWD,
    HLD,
    HOW,
    HPY,
    HUH,
    IDK,
    IFF,
    INS,
    MTO,
    NAR,
    NOT,
    OCC,
    ORR,
    PCE,
    POB,
    PRP,
    QRY,
    REJ,
    REM,
    ROF,
    RTO,
    SCD,
    SLO,
    SND,
    SRY,
    SUG,
    SUP,
    THK,
    TRY,
    UHY,
    ULB,
    UUB,
    WHT,
    WHY,
    WVE,
    XDO,
    XOY,
    YDO,
    YES,
    AnyDAIDEToken,
    Location,
    MoveByCVY,
    PowerAndSupplyCenters,
    Turn,
    Unit,
)

from chiron_utils.utils import POWER_NAMES_DICT, parse_daide

DaideObject = Union[AnyDAIDEToken, Location, PowerAndSupplyCenters, Turn, Unit, str]

power_list = list(POWER_NAMES_DICT)
unit_dict = {
    "FLT": "fleet",
    "AMY": "army",
}


def gen_english(
    daide: Union[AnyDAIDEToken, str],
    sender: str = "I",
    recipient: str = "You",
    *,
    make_natural: bool = True,
) -> str:
    """Generate English from DAIDE.

    If `make_natural` is true, first- and second-person pronouns/possessives will be used instead.
    We don't recommend passing in `make_natural=False` unless there is a specific reason to do so.

    Params:
        daide: DAIDE string, e.g., "(ENG FLT LON) BLD".
        sender: Power sending the message, e.g., "ENG".
        recipient: Power to which the message is sent, e.g., "TUR".
        make_natural: Whether to make the resulting English sound "natural".

    Returns:
        DAIDE message converted to English.
    """
    if not make_natural and (not sender or not recipient):
        return "ERROR: sender and recipient must be provided if make_natural is False"

    try:
        if isinstance(daide, str):
            parsed_daide = parse_daide(daide)
        else:
            parsed_daide = daide
        eng = daide_to_en(parsed_daide)
        return post_process(eng, sender, recipient, make_natural=make_natural)

    except ValueError as e:
        return "ERROR value: " + str(e)


def and_items(items: Sequence[DaideObject]) -> str:
    """Convert a list of items into an English list joined by "and"."""
    if len(items) == 1:
        return daide_to_en(items[0]) + " "
    elif len(items) == 2:
        return daide_to_en(items[0]) + " and " + daide_to_en(items[1]) + " "
    else:
        return (
            ", ".join([daide_to_en(item) for item in items[:-1]])
            + ", and "
            + daide_to_en(items[-1])
            + " "
        )


def or_items(items: Sequence[DaideObject]) -> str:
    """Convert a list of items into an English list joined by "or"."""
    if len(items) == 1:
        return daide_to_en(items[0]) + " "
    elif len(items) == 2:
        return daide_to_en(items[0]) + " or " + daide_to_en(items[1]) + " "
    else:
        return (
            ", ".join([daide_to_en(item) for item in items[:-1]])
            + ", or "
            + daide_to_en(items[-1])
            + " "
        )


def daide_to_en(daide: DaideObject) -> str:
    """Convert a DAIDE object (or raw string) into a mostly-English string."""
    if isinstance(daide, str):
        return daide

    # From `base_keywords.py`
    if isinstance(daide, Location):
        if daide.coast:
            return f"({daide.province} {daide.coast})"
        return cast(str, daide.province)
    if isinstance(daide, Unit):
        unit = unit_dict[daide.unit_type]
        return f"{daide.power}'s {unit} in {daide_to_en(daide.location)} "
    if isinstance(daide, HLD):
        return f"holding {daide_to_en(daide.unit)} "
    if isinstance(daide, MTO):
        return f"moving {daide_to_en(daide.unit)} to {daide_to_en(daide.location)} "
    if isinstance(daide, SUP):
        if not daide.province_no_coast:
            return f"using {daide_to_en(daide.supporting_unit)} to support {daide_to_en(daide.supported_unit)} "
        else:
            return f"using {daide_to_en(daide.supporting_unit)} to support {daide_to_en(daide.supported_unit)} moving into {daide.province_no_coast} "
    if isinstance(daide, CVY):
        return f"using {daide_to_en(daide.convoying_unit)} to convoy {daide_to_en(daide.convoyed_unit)} into {daide_to_en(daide.province)} "
    if isinstance(daide, MoveByCVY):
        return (
            f"moving {daide_to_en(daide.unit)} by convoy into {daide_to_en(daide.province)} via "
            + and_items([daide_to_en(x) for x in daide.province_seas])
        )
    if isinstance(daide, RTO):
        return f"retreating {daide_to_en(daide.unit)} to {daide_to_en(daide.location)} "
    if isinstance(daide, DSB):
        return f"disbanding {daide_to_en(daide.unit)} "
    if isinstance(daide, BLD):
        return f"building {daide_to_en(daide.unit)} "
    if isinstance(daide, REM):
        return f"removing {daide_to_en(daide.unit)} "
    if isinstance(daide, WVE):
        return f"waiving {daide.power} "
    if isinstance(daide, Turn):
        return f"{daide.season} {daide.year} "

    # From `press_keywords.py`
    if isinstance(daide, PCE):
        return "peace between " + and_items(daide.powers)
    if isinstance(daide, CCL):
        return f'cancel "{daide_to_en(daide.press_message)}" '
    if isinstance(daide, TRY):
        return "try the following tokens: " + " ".join(daide.try_tokens) + " "
    if isinstance(daide, HUH):
        return f'not understand "{daide_to_en(daide.press_message)}" '
    if isinstance(daide, PRP):
        return f"propose {daide_to_en(daide.arrangement)} "
    if isinstance(daide, ALYVSS):
        if not any(pp in daide.aly_powers for pp in daide.vss_powers):
            # if there is VSS power and no overlap between the allies and the enemies
            return (
                "an alliance with "
                + and_items(daide.aly_powers)
                + "against "
                + and_items(daide.vss_powers)
            )
        else:
            return "an alliance of " + and_items(daide.aly_powers)
    if isinstance(daide, SLO):
        return f"{daide.power} solo"
    if isinstance(daide, NOT):
        return f"not {daide_to_en(daide.arrangement_qry)} "
    if isinstance(daide, NAR):
        return f"lack of arrangement: {daide_to_en(daide.arrangement)} "
    if isinstance(daide, DRW):
        if daide.powers:
            return and_items(daide.powers) + "draw "
        else:
            return "draw"
    if isinstance(daide, YES):
        return f"accept {daide_to_en(daide.press_message)} "
    if isinstance(daide, REJ):
        return f"reject {daide_to_en(daide.press_message)} "
    if isinstance(daide, BWX):
        return f"refuse answering to {daide_to_en(daide.press_message)} "
    if isinstance(daide, FCT):
        return f'expect the following: "{daide_to_en(daide.arrangement_qry_not)}" '
    if isinstance(daide, FRM):
        return (
            f"from {daide.frm_power} to "
            + and_items(daide.recv_powers)
            + f': "{daide_to_en(daide.message)}" '
        )
    if isinstance(daide, XDO):
        return f"an order {daide_to_en(daide.order)} "
    if isinstance(daide, DMZ):
        return (
            and_items(daide.powers)
            + "demilitarize "
            + and_items([daide_to_en(x) for x in daide.provinces])
        )
    if isinstance(daide, AND):
        return and_items(daide.arrangements)
    if isinstance(daide, ORR):
        return or_items(daide.arrangements)
    if isinstance(daide, PowerAndSupplyCenters):
        return f"{daide.power} to have " + and_items([daide_to_en(x) for x in daide.supply_centers])
    if isinstance(daide, SCD):
        pas_str = [daide_to_en(pas) + " " for pas in daide.power_and_supply_centers]
        return "an arrangement of supply centre distribution as follows: " + and_items(pas_str)
    if isinstance(daide, OCC):
        unit_str = [daide_to_en(unit) for unit in daide.units]
        return "placing " + and_items(unit_str)
    if isinstance(daide, CHO):
        if daide.minimum == daide.maximum:
            return f"choosing {daide.minimum} in " + and_items(daide.arrangements)
        else:
            return f"choosing between {daide.minimum} and {daide.maximum} in " + and_items(
                daide.arrangements
            )
    if isinstance(daide, INS):
        return f"insist {daide_to_en(daide.arrangement)} "
    if isinstance(daide, QRY):
        return f"Is {daide_to_en(daide.arrangement)} true? "
    if isinstance(daide, THK):
        return f"think {daide_to_en(daide.arrangement_qry_not)} is true "
    if isinstance(daide, IDK):
        return f"don't know about {daide_to_en(daide.qry_exp_wht_prp_ins_sug)} "
    if isinstance(daide, SUG):
        return f"suggest {daide_to_en(daide.arrangement)} "
    if isinstance(daide, WHT):
        return f"What do you think about {daide_to_en(daide.unit)} ? "
    if isinstance(daide, HOW):
        return f"How do you think we should attack {daide.province_power} ? "
    if isinstance(daide, EXP):
        return f"The explanation for what they did in {daide_to_en(daide.turn)} is {daide_to_en(daide.message)} "
    if isinstance(daide, SRY):
        return f"I'm sorry about {daide_to_en(daide.exp)} "
    if isinstance(daide, FOR):
        if not daide.end_turn:
            return f"{daide_to_en(daide.arrangement)} in {daide_to_en(daide.start_turn)} "
        else:
            return f"{daide_to_en(daide.arrangement)} from {daide_to_en(daide.start_turn)} to {daide_to_en(daide.end_turn)} "
    if isinstance(daide, IFF):
        if not daide.els_press_message:
            return f'if {daide_to_en(daide.arrangement)} then "{daide_to_en(daide.press_message)}" '
        else:
            return f'if {daide_to_en(daide.arrangement)} then "{daide_to_en(daide.press_message)}" else "{daide_to_en(daide.els_press_message)}" '
    if isinstance(daide, XOY):
        return f"{daide.power_x} owes {daide.power_y} "
    if isinstance(daide, YDO):
        unit_str = [daide_to_en(unit) for unit in daide.units]
        return f"giving {daide.power} the control of" + and_items(unit_str)
    if isinstance(daide, SND):
        return f"{daide.power} sending {daide_to_en(daide.message)} to " + and_items(
            daide.recv_powers
        )
    if isinstance(daide, FWD):
        return (
            f"forwarding to {daide.power_2} if {daide.power_1} receives message from "
            + and_items(daide.powers)
        )
    if isinstance(daide, BCC):
        return f"forwarding to {daide.power_2} if {daide.power_1} sends message to " + and_items(
            daide.powers
        )
    if isinstance(daide, WHY):
        return f'Why do you believe "{daide_to_en(daide.fct_thk_prp_ins)}" ? '
    if isinstance(daide, POB):
        return f'answer "{daide_to_en(daide.why)}": the position on the board, or the previous moves, suggests/implies it '
    if isinstance(daide, UHY):
        return f'am unhappy that "{daide_to_en(daide.press_message)}" '
    if isinstance(daide, HPY):
        return f'am happy that "{daide_to_en(daide.press_message)}" '
    if isinstance(daide, ANG):
        return f'am angry that "{daide_to_en(daide.press_message)}" '
    if isinstance(daide, ROF):
        return "requesting an offer"
    if isinstance(daide, ULB):
        return f"having a utility lower bound of float for {daide.power} is {daide.float_val} "
    if isinstance(daide, UUB):
        return f"having a utility upper bound of float for {daide.power} is {daide.float_val} "

    raise ValueError(f"Unable to process {daide}")


def post_process(sentence: str, sender: str, recipient: str, *, make_natural: bool) -> str:
    """Make a sentence more grammatical and readable.

    Params:
        sentence: English string, e.g., "reject propose build fleet LON".
        sender: Power sending the message, e.g., "ENG".
        recipient: Power to which the message is sent, e.g., "TUR".
        make_natural: Whether to make the resulting English sound "natural".

    Returns:
        More readable English string.
    """
    # if sender or recipient is not provided, use first and second
    # person (default case).
    if make_natural:
        agent_subjective = "I"
        recipient_subjective = "you"
        recipient_possessive = "your"
        agent_objective = "me"
        recipient_objective = recipient_subjective

    else:
        agent_subjective = sender
        recipient_subjective = recipient
        recipient_possessive = recipient + "'s"
        agent_objective = sender
        recipient_objective = recipient

    output = sentence.replace("in <location>", "")
    output = output.replace("<country>'s", "")

    # general steps that apply to all types of daide messages
    output = agent_subjective + " " + output

    # remove extra spaces
    output = " ".join(output.split())

    # add period if needed
    if not output.endswith(".") or not output.endswith("?"):
        output += "."

    # substitute power names with pronouns
    if make_natural:
        output = output.replace(" " + sender + " ", " " + agent_objective + " ")
        output = output.replace(" " + recipient + " ", " " + recipient_objective + " ")

    # case-dependent handling

    # REJ/YES
    if "reject" in output or "accept" in output:
        output = output.replace("propose", recipient_possessive + " proposal of", 1)

    # make natural for proposals
    detect_str = f"I propose an order using {sender}'s"
    if sender != "I" and make_natural and detect_str in output:
        output = output.replace(detect_str, "I will move")
    elif sender in power_list and make_natural:
        output = output.replace("I propose an order using", "I think")
        output = output.replace(" to ", " is going to ")
    return output
