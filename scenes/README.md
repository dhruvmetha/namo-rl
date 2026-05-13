# scenes/

MuJoCo scene XMLs for the diff-drive car. Each XML is fully self-contained — no `<include>` directives, no external mesh files, only built-in MuJoCo textures.

Each scene contains:

- One body `walls` with one or more `wall_*` box geoms (walls inflict episode termination if the car touches them).
- Zero or more `obstacle_*_movable` bodies with `<joint type="free"/>` (the car can shove these).
- One `<site name="goal">` whose `pos` is the target position (XY).
- The diff-drive car: body `car` with sub-bodies `left_wheel`, `right_wheel` and actuators `left_wheel_drive`, `right_wheel_drive`.

## Provenance

Copied from `/common/home/dm1487/robotics_research/ktamp/namo` @ commit `1361d42` on branch `car-baseline`.

- `nav_env.xml` ← `test_xml/little-car-modeling-package/artifacts/nav_env.xml`
- `nav_env_3000e.xml` ← `test_xml/little-car-modeling-package/artifacts/nav_env_3000e.xml`

Add more scenes by dropping `.xml` files into `car/`. The env loader picks them up on next instantiation.
