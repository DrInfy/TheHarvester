# Drones are in supply workers
from sc2 import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

zerg_upgrades = [
    UpgradeId.ZERGLINGMOVEMENTSPEED,
    UpgradeId.CENTRIFICALHOOKS,
    UpgradeId.ZERGLINGATTACKSPEED,
    UpgradeId.CHITINOUSPLATING,
    UpgradeId.ANABOLICSYNTHESIS,
    UpgradeId.GLIALRECONSTITUTION,
    UpgradeId.EVOLVEMUSCULARAUGMENTS,
    UpgradeId.EVOLVEGROOVEDSPINES,
    UpgradeId.BURROW,
    UpgradeId.NEURALPARASITE,
]


protoss_upgrades = [
    [UpgradeId.PROTOSSAIRARMORSLEVEL1, UpgradeId.PROTOSSAIRARMORSLEVEL2, UpgradeId.PROTOSSAIRARMORSLEVEL3],
    [UpgradeId.PROTOSSAIRWEAPONSLEVEL1, UpgradeId.PROTOSSAIRWEAPONSLEVEL2, UpgradeId.PROTOSSAIRWEAPONSLEVEL3],
    [UpgradeId.PROTOSSGROUNDARMORSLEVEL1, UpgradeId.PROTOSSGROUNDARMORSLEVEL2, UpgradeId.PROTOSSGROUNDARMORSLEVEL3],
    [UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1, UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2, UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3],
    [UpgradeId.PROTOSSSHIELDSLEVEL1, UpgradeId.PROTOSSSHIELDSLEVEL2, UpgradeId.PROTOSSSHIELDSLEVEL3],
    UpgradeId.ADEPTPIERCINGATTACK,
    UpgradeId.WARPGATERESEARCH,
    UpgradeId.DARKTEMPLARBLINKUPGRADE,
    UpgradeId.PHOENIXRANGEUPGRADE,
    UpgradeId.VOIDRAYSPEEDUPGRADE,
    UpgradeId.EXTENDEDTHERMALLANCE,
    UpgradeId.GRAVITICDRIVE,
    UpgradeId.OBSERVERGRAVITICBOOSTER,
    UpgradeId.PSISTORMTECH,
    UpgradeId.ADEPTPIERCINGATTACK,
    UpgradeId.BLINKTECH,
    UpgradeId.CHARGE,
]

zerg_units = [
    UnitTypeId.LARVA,
    UnitTypeId.QUEEN,
    UnitTypeId.ZERGLING,
    UnitTypeId.BANELINGCOCOON,
    UnitTypeId.BANELING,
    UnitTypeId.ROACH,
    UnitTypeId.RAVAGER,
    UnitTypeId.RAVAGERCOCOON,
    UnitTypeId.HYDRALISK,
    UnitTypeId.LURKERMP,
    UnitTypeId.INFESTOR,
    UnitTypeId.SWARMHOSTMP,
    UnitTypeId.ULTRALISK,
    UnitTypeId.OVERLORD,
    UnitTypeId.OVERLORDTRANSPORT,
    UnitTypeId.OVERSEER,
    UnitTypeId.CHANGELING,
    UnitTypeId.MUTALISK,
    UnitTypeId.CORRUPTOR,
    UnitTypeId.VIPER,
    UnitTypeId.BROODLORD,
    UnitTypeId.BROODLORDCOCOON,
]

terran_units = [
    UnitTypeId.MARINE,
    UnitTypeId.MARAUDER,
    UnitTypeId.REAPER,
    UnitTypeId.GHOST,
    UnitTypeId.HELLION,
    UnitTypeId.WIDOWMINE,
    UnitTypeId.SIEGETANK,
    UnitTypeId.CYCLONE,
    UnitTypeId.THOR,
    UnitTypeId.VIKINGFIGHTER,
    UnitTypeId.MEDIVAC,
    UnitTypeId.LIBERATOR,
    UnitTypeId.BANSHEE,
    UnitTypeId.RAVEN,
    UnitTypeId.BATTLECRUISER,
]

protoss_units = [
    UnitTypeId.ZEALOT,
    UnitTypeId.SENTRY,
    UnitTypeId.STALKER,
    UnitTypeId.ADEPT,
    UnitTypeId.HIGHTEMPLAR,
    UnitTypeId.DARKTEMPLAR,
    UnitTypeId.ARCHON,
    UnitTypeId.OBSERVER,
    UnitTypeId.WARPPRISM,
    UnitTypeId.IMMORTAL,
    UnitTypeId.COLOSSUS,
    UnitTypeId.DISRUPTOR,
    UnitTypeId.PHOENIX,
    UnitTypeId.VOIDRAY,
    UnitTypeId.ORACLE,
    UnitTypeId.TEMPEST,
    UnitTypeId.CARRIER,
    UnitTypeId.INTERCEPTOR,
    UnitTypeId.MOTHERSHIP,
]

zerg_buildings = [
    UnitTypeId.HATCHERY,
    UnitTypeId.SPAWNINGPOOL,
    UnitTypeId.EVOLUTIONCHAMBER,
    UnitTypeId.SPINECRAWLER,
    UnitTypeId.SPORECRAWLER,
    UnitTypeId.ROACHWARREN,
    UnitTypeId.BANELINGNEST,
    UnitTypeId.LAIR,
    UnitTypeId.HYDRALISKDEN,
    UnitTypeId.LURKERDENMP,
    UnitTypeId.INFESTATIONPIT,
    UnitTypeId.SPIRE,
    UnitTypeId.NYDUSNETWORK,
    UnitTypeId.NYDUSCANAL,
    UnitTypeId.HIVE,
    UnitTypeId.ULTRALISKCAVERN,
    UnitTypeId.GREATERSPIRE,
]

terran_buildings = [
    UnitTypeId.COMMANDCENTER,
    UnitTypeId.ORBITALCOMMAND,
    UnitTypeId.PLANETARYFORTRESS,
    UnitTypeId.SUPPLYDEPOT,
    UnitTypeId.REFINERY,
    UnitTypeId.BARRACKS,
    UnitTypeId.ENGINEERINGBAY,
    UnitTypeId.BUNKER,
    UnitTypeId.MISSILETURRET,
    UnitTypeId.SENSORTOWER,
    UnitTypeId.FACTORY,
    UnitTypeId.GHOSTACADEMY,
    UnitTypeId.ARMORY,
    UnitTypeId.STARPORT,
    UnitTypeId.FUSIONCORE,
    UnitTypeId.TECHLAB,
    UnitTypeId.BARRACKSTECHLAB,
    UnitTypeId.FACTORYTECHLAB,
    UnitTypeId.STARPORTTECHLAB,
    UnitTypeId.REACTOR,
    UnitTypeId.BARRACKSREACTOR,
    UnitTypeId.FACTORYREACTOR,
    UnitTypeId.STARPORTREACTOR,
]

protoss_buildings = [
    UnitTypeId.NEXUS,
    UnitTypeId.PYLON,
    UnitTypeId.ASSIMILATOR,
    UnitTypeId.GATEWAY,
    UnitTypeId.FORGE,
    UnitTypeId.PHOTONCANNON,
    UnitTypeId.SHIELDBATTERY,
    UnitTypeId.WARPGATE,
    UnitTypeId.CYBERNETICSCORE,
    UnitTypeId.TWILIGHTCOUNCIL,
    UnitTypeId.ROBOTICSFACILITY,
    UnitTypeId.STARGATE,
    UnitTypeId.TEMPLARARCHIVE,
    UnitTypeId.DARKSHRINE,
    UnitTypeId.ROBOTICSBAY,
    UnitTypeId.FLEETBEACON,
]