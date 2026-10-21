"""Expanded development conditions and a separately reserved confirmation suite.

Version 1 remains available to reproduce the earlier candidate. Previously
inspected double-pulse and triangle tests are development families in version 2.
"""

from dataclasses import replace

from research_protocol import Condition


EXPANDED_SEEDS = (7109, 7211, 7307, 7411, 7523)
EXPANDED_NAMESPACE = 33
EXPANDED_WITHHELD = ("chirp", "exponential", "cosine_burst")
EXPANDED_PROFILES = ("constant", "pulse", "ramp", "sine", "ramp_down", "triangle", "double_pulse",
                     "random_pulses", "random_steps", "random_smooth", "random_ramp", "random_triangle", "random_sine")


def expanded_training_conditions(original):
    """Retain original controls and cross new profiles with observation noise."""
    conditions = list(original)
    for profile in EXPANDED_PROFILES[4:]:
        for index, (noise, jitter) in enumerate(((.2, .03), (5., .06), (10., .06))):
            conditions.append(Condition(f"{profile}_{index}", f"fit_{profile}", noise, noise, jitter, profile=profile))
    for profile in ("triangle", "random_pulses", "random_steps", "random_smooth"):
        conditions.append(Condition(f"{profile}_network", "fit_network", .5, 2., .06,
                                    profile=profile, network_weight=.5))
    return conditions


def expanded_comparison_conditions(config, original):
    conditions = []
    for condition in original:
        if condition.profile in ("triangle", "double_pulse"):
            condition = replace(condition, axis="familiar_stimulus", held_out_reason=None)
        conditions.append(condition)
    for profile in ("ramp_down", "random_pulses", "random_steps", "random_smooth"):
        conditions.append(Condition(profile, "familiar_stimulus", 0, config.observation_noise_std,
                                    config.parameter_jitter, profile=profile))
    for profile in EXPANDED_WITHHELD:
        for noise in (config.observation_noise_std, 5.):
            conditions.append(Condition(f"{profile}_noise_{noise:g}", "withheld_family", noise, noise,
                                        config.parameter_jitter, profile=profile,
                                        held_out_reason="family reserved before version-2 training; absent from fit, selection and calibration"))
    return conditions
