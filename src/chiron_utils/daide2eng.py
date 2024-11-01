from daidepp import ALYVSS, AND, ANG, AnyDAIDEToken, BCC, BLD, BWX, CCL, CHO, CVY, DMZ, DRW, DSB, EXP, FCT, FOR, FRM, \
    FWD, HLD, HOW, HPY, HUH, IDK, IFF, INS, Location, MTO, MoveByCVY, NAR, NOT, OCC, ORR, PCE, POB, PRP, \
    PowerAndSupplyCenters, QRY, REJ, REM, ROF, RTO, SCD, SLO, SND, SRY, SUG, SUP, THK, TRY, Turn, UHY, ULB, UUB, Unit, \
    WHT, WHY, WVE, XDO, XOY, YDO, YES

from chiron_utils.utils import POWER_NAMES_DICT, parse_daide

power_list = list(POWER_NAMES_DICT)
unit_dict = {
    "FLT": "fleet",
    "AMY": "army",
}


def gen_English(daide: str, sender="I", recipient="You", make_natural=True) -> str:
    '''
    Generate English from DAIDE. If make_natural is true, first and
    second person pronouns/possessives will be used instead. We don't
    recommend passing in make_natural=False unless there is a
    specific reason to do so.

    :param daide: DAIDE string, e.g. '(ENG FLT LON) BLD'
    :param sender: power sending the message, e.g., 'ENG'
    :param recipient: power to which the message is sent, e.g., 'TUR'
    '''

    if not make_natural and (not sender or not recipient):
        return "ERROR: sender and recipient must be provided if make_natural is False"

    try:
        parsed_daide = parse_daide(daide)
        eng = daide_to_en(parsed_daide)
        return post_process(eng, sender, recipient, make_natural)

    except ValueError as e:
        return "ERROR value: " + str(e)


def and_items(items):
    if len(items) == 1:
        return str(items[0]) + " "
    elif len(items) == 2:
        return str(items[0]) + " and " + str(items[1]) + " "
    else:
        return ", ".join([str(item) for item in items[:-1]]) + ", and " + str(items[-1]) + " "


def or_items(items):
    if len(items) == 1:
        return str(items[0]) + " "
    elif len(items) == 2:
        return str(items[0]) + " or " + str(items[1]) + " "
    else:
        return ", ".join([str(item) for item in items[:-1]]) + ", or " + str(items[-1]) + " "


def daide_to_en(daide: AnyDAIDEToken) -> str:
    # From `base_keywords.py`
    if isinstance(daide, Location):
        if daide.coast:
            return f"({daide.province} {daide.coast})"
        return daide.province
    if isinstance(daide, Unit):
        unit = unit_dict[daide.unit_type]
        return f"{daide.power}'s {unit} in {daide.location} "
    if isinstance(daide, HLD):
        return f"holding {daide.unit} "
    if isinstance(daide, MTO):
        return f"moving {daide.unit} to {daide.location} "
    if isinstance(daide, SUP):
        if not daide.province_no_coast:
            return f"using {daide.supporting_unit} to support {daide.supported_unit} "
        else:
            return f"using {daide.supporting_unit} to support {daide.supported_unit} moving into {daide.province_no_coast} "
    if isinstance(daide, CVY):
        return f"using {daide.convoying_unit} to convoy {daide.convoyed_unit} into {daide.province} "
    if isinstance(daide, MoveByCVY):
        return (
            f"moving {daide.unit} by convoy into {daide.province} via "
            + and_items(list(map(lambda x: str(x), daide.province_seas)))
        )
    if isinstance(daide, RTO):
        return f"retreating {daide.unit} to {daide.location} "
    if isinstance(daide, DSB):
        return f"disbanding {daide.unit} "
    if isinstance(daide, BLD):
        return f"building {daide.unit} "
    if isinstance(daide, REM):
        return f"removing {daide.unit} "
    if isinstance(daide, WVE):
        return f"waiving {daide.power} "
    if isinstance(daide, Turn):
        return f"{daide.season} {daide.year} "

    # From `press_keywords.py`
    if isinstance(daide, PCE):
        return "peace between " + and_items(daide.powers)
    if isinstance(daide, CCL):
        return f"cancel \"{daide.press_message}\" "
    if isinstance(daide, TRY):
        return "try the following tokens: " + " ".join(daide.try_tokens) + " "
    if isinstance(daide, HUH):
        return f"not understand \"{daide.press_message}\" "
    if isinstance(daide, PRP):
        return f"propose {daide.arrangement} "
    if isinstance(daide, ALYONLY):
        return "an alliance of " + and_items(daide.powers)
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
        return f"not {daide.arrangement_qry} "
    if isinstance(daide, NAR):
        return f"lack of arragement: {daide.arrangement} "
    if isinstance(daide, DRW):
        if daide.powers:
            return and_items(daide.powers) + "draw "
        else:
            return f"draw"
    if isinstance(daide, YES):
        return f"accept {daide.press_message} "
    if isinstance(daide, REJ):
        return f"reject {daide.press_message} "
    if isinstance(daide, BWX):
        return f"refuse answering to {daide.press_message} "
    if isinstance(daide, FCT):
        return f"expect the following: \"{daide.arrangement_qry_not}\" "
    if isinstance(daide, FRM):
        return (
            f"from {daide.frm_power} to "
            + and_items(daide.recv_powers)
            + f": \"{daide.message}\" "
        )
    if isinstance(daide, XDO):
        return f"an order {daide.order} "
    if isinstance(daide, DMZ):
        return (
            and_items(daide.powers)
            + "demilitarize "
            + and_items(list(map(lambda x: str(x), daide.provinces)))
        )
    if isinstance(daide, AND):
        return and_items(daide.arrangements)
    if isinstance(daide, ORR):
        return or_items(daide.arrangements)
    if isinstance(daide, PowerAndSupplyCenters):
        return f"{daide.power} to have " + and_items(list(map(lambda x: str(x), daide.supply_centers)))
    if isinstance(daide, SCD):
        pas_str = [str(pas) + " " for pas in daide.power_and_supply_centers]
        return f"an arragement of supply centre distribution as follows: " + and_items(pas_str)
    if isinstance(daide, OCC):
        unit_str = [str(unit) for unit in daide.units]
        return f"placing " + and_items(unit_str)
    if isinstance(daide, CHO):
        if daide.minimum == daide.maximum:
            return f"choosing {daide.minimum} in " + and_items(daide.arrangements)
        else:
            return f"choosing between {daide.minimum} and {daide.maximum} in " + and_items(daide.arrangements)
    if isinstance(daide, INS):
        return f"insist {daide.arrangement} "
    if isinstance(daide, QRY):
        return f"Is {daide.arrangement} true? "
    if isinstance(daide, THK):
        return f"think {daide.arrangement_qry_not} is true "
    if isinstance(daide, IDK):
        return f"don't know about {daide.qry_exp_wht_prp_ins_sug} "
    if isinstance(daide, SUG):
        return f"suggest {daide.arrangement} "
    if isinstance(daide, WHT):
        return f"What do you think about {daide.unit} ? "
    if isinstance(daide, HOW):
        return f"How do you think we should attack {daide.province_power} ? "
    if isinstance(daide, EXP):
        return f"The explanation for what {daide.power} did in {daide.turn} is {daide.message} "
    if isinstance(daide, SRY):
        return f"I'm sorry about {daide.exp} "
    if isinstance(daide, FOR):
        if not daide.end_turn:
            return f"{daide.arrangement} in {daide.start_turn} "
        else:
            return f"{daide.arrangement} from {daide.start_turn} to {daide.end_turn} "
    if isinstance(daide, IFF):
        if not daide.els_press_message:
            return f"if {daide.arrangement} then \"{daide.press_message}\" "
        else:
            return f"if {daide.arrangement} then \"{daide.press_message}\" else \"{daide.els_press_message}\" "
    if isinstance(daide, XOY):
        return f"{daide.power_x} owes {daide.power_y} "
    if isinstance(daide, YDO):
        unit_str = [str(unit) for unit in daide.units]
        return f"giving {daide.power} the control of" + and_items(unit_str)
    if isinstance(daide, SND):
        return (
            f"{daide.power} sending {daide.message} to "
            + and_items(daide.recv_powers)
        )
    if isinstance(daide, FWD):
        return (
            f"forwarding to {daide.power_2} if {daide.power_1} receives message from "
            + and_items(daide.powers)
        )
    if isinstance(daide, BCC):
        return (
            f"forwarding to {daide.power_2} if {daide.power_1} sends message to "
            + and_items(daide.powers)
        )
    if isinstance(daide, WHY):
        return f"Why do you believe \"{daide.fct_thk_prp_ins}\" ? "
    if isinstance(daide, POB):
        return f"answer \"{daide.why}\": the position on the board, or the previous moves, suggests/implies it "
    if isinstance(daide, UHY):
        return f"am unhappy that \"{daide.press_message}\" "
    if isinstance(daide, HPY):
        return f"am happy that \"{daide.press_message}\" "
    if isinstance(daide, ANG):
        return f"am angry that \"{daide.press_message}\" "
    if isinstance(daide, ROF):
        return f"requesting an offer"
    if isinstance(daide, ULB):
        return f"having a utility lower bound of float for {daide.power} is {daide.float_val} "
    if isinstance(daide, UUB):
        return f"having a utility upper bound of float for {daide.power} is {daide.float_val} "

    return str(daide)


def post_process(sentence: str, sender: str, recipient: str, make_natural: bool) -> str:
    '''
    Make the sentence more grammatical and readable
    :param sentence: string, e.g. 'reject propose build fleet LON'
    '''

    # if sender or recipient is not provided, use first and second
    # person (default case).
    if make_natural:
        AGENT_SUBJECTIVE = 'I'
        RECIPIENT_SUBJECTIVE = 'you'
        AGENT_POSSESSIVE = 'my'
        RECIPIENT_POSSESSIVE = 'your'
        AGENT_OBJECTIVE = 'me'
        RECIPIENT_OBJECTIVE = RECIPIENT_SUBJECTIVE

    else:
        AGENT_SUBJECTIVE = sender
        RECIPIENT_SUBJECTIVE = recipient
        AGENT_POSSESSIVE = sender + "'s"
        RECIPIENT_POSSESSIVE = recipient + "'s"
        AGENT_OBJECTIVE = sender
        RECIPIENT_OBJECTIVE = recipient

    output = sentence.replace("in <location>", "")
    output = output.replace("<country>'s", "")

    # general steps that apply to all types of daide messages
    output = AGENT_SUBJECTIVE + ' ' + output

    # remove extra spaces
    output = " ".join(output.split())

    # add period if needed
    if not output.endswith('.') or not output.endswith('?'):
        output += '.'

    # substitute power names with pronouns
    if make_natural:
        output = output.replace(' ' + sender + ' ', ' ' + AGENT_OBJECTIVE + ' ')
        output = output.replace(' ' + recipient + ' ', ' ' + RECIPIENT_OBJECTIVE + ' ')

    # case-dependent handling

    # REJ/YES
    if "reject" in output or "accept" in output:
        output = output.replace(
            'propose', RECIPIENT_POSSESSIVE + ' proposal of', 1)

    # make natural for proposals
    detect_str = f"I propose an order using {sender}'s"
    if sender != "I" and make_natural and detect_str in output:
        output = output.replace(detect_str, f"I will move")
    elif sender in power_list and make_natural:
        output = output.replace('I propose an order using', 'I think')
        output = output.replace(' to ', ' is going to ')
    return output
