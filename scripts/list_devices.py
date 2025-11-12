import sys
import sounddevice as sd

# SELECT DEVICES
# export INPUT_DEV=<index>    # mic index from your listing
# export OUTPUT_DEV=<index>   # speaker/headphones index

def main():
    try:
        print("=== Host APIs ===")
        for i, api in enumerate(sd.query_hostapis()):
            print(f"[{i}] {api['name']}")
        print()

        hostapis = sd.query_hostapis()
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]

        print("=== Audio Devices ===")
        has_input, has_output = False, False
        for idx, dev in enumerate(devices):
            input_flag = "IN" if dev["max_input_channels"] > 0 else ""
            output_flag = "OUT" if dev["max_output_channels"] > 0 else ""
            flags = ",".join(filter(None, [input_flag, output_flag]))
            host = hostapis[dev["hostapi"]]["name"]

            default_marker = ""
            if idx == default_input:
                default_marker += " [default input]"
                has_input = True
            if idx == default_output:
                default_marker += " [default output]"
                has_output = True

            print(f"[{idx:2d}] {dev['name']} — {flags}{default_marker} — host={host}")

        if not has_input or not has_output:
            print("Error: No valid input or output devices detected.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error listing devices: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()