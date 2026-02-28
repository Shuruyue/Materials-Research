
from mace.calculators import mace_mp


def inspect_model():
    print("Loading MACE model...")
    # Load small model for speed, assuming structure is consistent across sizes
    # Or load 'medium' since that's what we default to
    pretrained_mace = mace_mp(model="medium", device="cpu", default_dtype="float32").models[0]

    print("\n--- Model Attributes ---")
    print(f"r_max: {pretrained_mace.r_max}")
    print("\n--- Interactions[0] ---")
    if len(pretrained_mace.interactions) > 0:
        inter = pretrained_mace.interactions[0]
        print(f"Interaction Class: {type(inter)}")
        if hasattr(inter, 'correlation'):
            print(f"Found correlation: {getattr(inter, 'correlation')}")
        else:
            print("Correlation attribute NOT found in interaction.")

    print("\n--- Readouts[0] ---")
    if len(pretrained_mace.readouts) > 0:
        readout = pretrained_mace.readouts[0]
        print(f"Readout Class: {type(readout)}")
        print(f"Readout Dir: {dir(readout)}")
        if hasattr(readout, 'hidden_irreps'):
            print(f"Found hidden_irreps: {getattr(readout, 'hidden_irreps')}")
        if hasattr(readout, 'irreps_in'):
            print(f"Found irreps_in: {getattr(readout, 'irreps_in')}")
        if hasattr(readout, 'irreps_out'):
            print(f"Found irreps_out: {getattr(readout, 'irreps_out')}")

    print("\n--- Radial Basis ---")
    bessel = pretrained_mace.radial_embedding.bessel_fn
    print(f"Bessel Dir: {dir(bessel)}")
    if hasattr(bessel, 'number_of_basis'):
        print(f"Found number_of_basis: {bessel.number_of_basis}")
    if hasattr(bessel, 'num_basis'):
        print(f"Found num_basis: {bessel.num_basis}")


if __name__ == "__main__":
    inspect_model()
