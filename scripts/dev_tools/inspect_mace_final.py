
from mace.calculators import mace_mp


def inspect_model():
    print("Loading MACE model...")
    pretrained_mace = mace_mp(model="medium", device="cpu", default_dtype="float32").models[0]

    print("\n--- Interactions[0] Dir ---")
    if len(pretrained_mace.interactions) > 0:
        inter = pretrained_mace.interactions[0]
        print(dir(inter))

if __name__ == "__main__":
    inspect_model()
