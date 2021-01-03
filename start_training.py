import subprocess

wsl = "wsl python3.7 /mnt/" + YOUR_PATH_TO_HARVESTER
# to = "--timeout 900 -z"
to = "-p2 ai.terran.hard"
to2 = "-p2 ai.zerg.hard"
to3 = "-p2 ai.protoss.hard"

def ai_opponents(difficulty: str) -> str:
    text = ""
    for race in ["zerg", "protoss", "terran"]:
        for build in ["rush", "timing", "power", "air", "air", "macro"]:
            text += f"ai.{race}.{difficulty}.{build},"
    return text.strip(",")


harvester_test_pattern = (
    "harvesterzerg.learning,"
    "harvesterzerg.scripted,"
    "harvesterzerg.scripted.default.2,"
    "harvesterzerg.learning.default.2,"
    "harvesterzerg.scripted.default.3,"
    "harvesterzerg.learning.default.3,"
    "harvesterzerg.scripted.default.4,"
    "harvesterzerg.learning.default.4,"
    "harvesterzerg.scripted.default.5,"
    "harvesterzerg.learning.default.5,"
    "harvesterzerg.scripted.default.6,"
    "harvesterzerg.learning.default.6,"
    "harvesterzerg.scripted.default.7,"
    "harvesterzerg.learning.default.7,"
    "harvesterzerg.play.default.master,"
).strip(",")
cmd_list_ml = [
    # f"{wsl} -p1 harvesterzerg.learning -p2 harvesterzerg.learning.default.2",
    # f"{wsl} -p1 harvesterzerg.learning.default.2 -p2 harvesterzerg.learning.default.3",
    # f"{wsl} -p1 harvesterzerg.learning -p2 harvesterzerg.learning.default.3",
]
for i in range(0, 15):
    cmd_list_ml.append(f'{wsl} -p1 {harvester_test_pattern} -p2 {ai_opponents("hard")}')

for i in range(0, 15):
    cmd_list_ml.append(f'{wsl} -p1 {harvester_test_pattern} -p2 {ai_opponents("veryhard")}')


index = 0
for cmd in cmd_list_ml:
    index += 1
    final_cmd = cmd + " --port " + str(10000 + index * 10)
    cmds = final_cmd.split(" ")
    subprocess.Popen(cmds)
