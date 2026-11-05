"""Prime groups and scale targets for the low-basis encapsulation example.

The main QP satisfies the configured modulus-budget check. The example does
not assign a security level to fixed-weight secrets or the joint evaluation-key
transcript. Source Q has two primes; CTS/reduction/STC consume 33 of 51 Q rows.
"""

import math
from fhelium import CkksConfig


def parameters():
    q_groups = (
        (254279681, 253493249),
        (253100033, 249561089),
        (246415361, 245760001),
        (245235713, 244973569, 244842497),
        (241827841, 240648193),
        (239861761, 239337473),
        (236716033, 236584961),
        (235798529, 234356737),
        (232652801, 232390657),
        (231473153, 230817793),
        (230686721, 230424577),
        (228720641, 228458497),
        (228065281, 225574913),
        (224133121, 224002049),
        (223215617,),
        (222953473,),
        (222167041, 222035969),
        (221249537, 220594177),
        (219676673, 218628097),
        (217317377, 216924161),
        (216137729, 215744513),
        (215482369, 214171649),
        (211812353, 211025921),
        (210370561, 209059841),
        (207880193, 206307329),
        (205651969, 204865537),
    )
    p_moduli = (
        257949697,
        258605057,
        260571137,
        260702209,
        261488641,
        261881857,
        263323649,
        263454721,
        264634369,
        265420801,
        268042241,
    )
    delta = math.prod(q_groups[-1]) / 32
    config = CkksConfig(
        logN=16,
        q_depth_groups=q_groups,
        p_moduli=p_moduli,
        default_scale=delta,
        galois_generator=5,
    )
    auxiliary = dict(
        low_p_moduli=(268042241,),
        return_prime=256376833,
        return_p_moduli=(265420801, 268042241),
    )
    scales = [2.0**55] * (config.max_depth + 1)
    for depth in reversed(range(5, 14)):
        scales[depth] = math.sqrt(
            config.rescale_divisor(depth) * scales[depth + 1]
        )
    scales[17] = delta
    return config, auxiliary, tuple(scales)
