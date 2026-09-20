import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mission.active_sensing import (
    get_active_sensing_controller,
)
from src.mission.uav_controller import (
    UAVPosition,
    get_uav_controller,
)


def print_position(label, position):
    print(
        f"{label}: "
        f"X={position.x:.2f} m | "
        f"Y={position.y:.2f} m | "
        f"Z={position.z:.2f} m"
    )


def main():

    print("=" * 70)
    print("SENSE-X — CLOSED-LOOP ACTIVE SENSING TEST")
    print("=" * 70)

    controller = get_active_sensing_controller()
    uav = get_uav_controller()

    # Initial UAV position
    position = UAVPosition(
        x=0.0,
        y=0.0,
        z=10.0,
    )

    print_position("Initial UAV position", position)

    # ---------------------------------------------------------
    # OBSERVATION 1
    # ---------------------------------------------------------
    print("\n[1] INITIAL OBSERVATION")

    fusion_score = 0.56
    uncertainty = 0.38
    localization_confidence = 0.54
    priority_level = "MEDIUM"

    print(f"Fusion score:              {fusion_score:.4f}")
    print(f"Uncertainty:               {uncertainty:.4f}")
    print(
        f"Localization confidence:   "
        f"{localization_confidence:.4f}"
    )

    decision = controller.decide(
        fusion_score=fusion_score,
        uncertainty=uncertainty,
        priority_level=priority_level,
        localization_confidence=localization_confidence,
    )

    print(f"Decision:                  {decision.action}")
    print(f"Reason:                    {decision.reason}")

    # ---------------------------------------------------------
    # ACTIVE RESPONSE
    # ---------------------------------------------------------
    if decision.action == "REPOSITION_AND_RESCAN":

        print("\n[2] ACTIVE RESPONSE")
        print("Uncertainty is high.")
        print("Repositioning UAV for another observation...")

        movement = uav.reposition(
            position,
            direction="lateral",
        )

        position = movement.new_position

        print_position(
            "New UAV position",
            position,
        )

        print(
            f"Movement distance:         "
            f"{movement.movement_distance:.2f} m"
        )

    # ---------------------------------------------------------
    # OBSERVATION 2
    # ---------------------------------------------------------
    print("\n[3] SECOND OBSERVATION")

    # Simulated improvement after repositioning.
    second_fusion_score = 0.82
    second_uncertainty = 0.16
    second_localization_confidence = 0.87
    second_priority_level = "HIGH"

    print(
        f"Fusion score:              "
        f"{second_fusion_score:.4f}"
    )

    print(
        f"Uncertainty:               "
        f"{second_uncertainty:.4f}"
    )

    print(
        f"Localization confidence:   "
        f"{second_localization_confidence:.4f}"
    )

    second_decision = controller.decide(
        fusion_score=second_fusion_score,
        uncertainty=second_uncertainty,
        priority_level=second_priority_level,
        localization_confidence=second_localization_confidence,
    )

    print(
        f"Decision:                  "
        f"{second_decision.action}"
    )

    print(
        f"Reason:                    "
        f"{second_decision.reason}"
    )

    # ---------------------------------------------------------
    # FINAL STATE
    # ---------------------------------------------------------
    print("\n[4] CLOSED-LOOP RESULT")

    print("Initial state:")
    print("  → uncertain human evidence")

    print("Action:")
    print("  → UAV repositioned")

    print("Second observation:")
    print("  → stronger multimodal evidence")

    print("Final mission decision:")
    print(
        f"  → {second_decision.action}"
    )

    print("\n" + "=" * 70)
    print("ACTIVE SENSING LOOP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()