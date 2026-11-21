import subprocess
import re


def get_ip_address():
    result = subprocess.run(
        ["ipconfig"], capture_output=True, text=True, encoding="cp932"
    )
    lines = result.stdout.split("\n")

    wifi_section = False
    for line in lines:
        if "Wireless LAN adapter Wi-Fi:" in line:
            wifi_section = True
        elif wifi_section and ("IPv4 アドレス" in line or "IPv4 Address" in line):
            match = re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", line)
            if match:
                ip_address = match.group(0)
                return ip_address


if __name__ == "__main__":
    ip_address = get_ip_address()
    print("===============================")
    print(f"YOUR IP ADDRESS: {ip_address}")
    print("===============================")
