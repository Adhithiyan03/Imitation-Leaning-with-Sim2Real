To transform the long bumper, we have to do the following transformations:

```python
import FreeCAD as App
import math

# Get the document
doc = App.ActiveDocument

# Get the object by its label
obj = None
for o in doc.Objects:
    if o.Label == '3035571_LONG_BUMPER':
        obj = o
        break

if obj is None:
    print("Object with label '3035571_LONG_BUMPER' not found.")
    exit()

# Define the angles for rotation (in degrees)
angle_y = 180  # Rotate 180 degrees about the y-axis
angle_z = -95.296  # Rotate 95.296 degrees about the z-axis

# Create rotation objects
rotation_y = App.Rotation(App.Vector(0, 1, 0), angle_y)
rotation_z = App.Rotation(App.Vector(0, 0, 1), angle_z)

# Combine the rotations
combined_rotation = rotation_y.multiply(rotation_z)

# Set the placement of the object
obj.Placement = App.Placement(obj.Placement.Base, combined_rotation)

# Re-compute the document
doc.recompute()

```

Now for the switch cam, we have to do the following transformations:

```python
import FreeCAD as App
import math

# Get the document
doc = App.ActiveDocument

# Get the object by its label
obj = None
for o in doc.Objects:
    if o.Label == '3031075_SWITCH_CAM':
        obj = o
        break

if obj is None:
    print("Object with label '3035571_LONG_BUMPER' not found.")
    exit()

# Define the angles for rotation (in degrees)
angle_y = 90  # Rotate 180 degrees about the y-axis
angle_z = -12.404  # Rotate 95.296 degrees about the z-axis

# Create rotation objects
rotation_y = App.Rotation(App.Vector(0, 1, 0), angle_y)
rotation_z = App.Rotation(App.Vector(0, 0, 1), angle_z)

# Combine the rotations
combined_rotation = rotation_y.multiply(rotation_z)

# Set the placement of the object
obj.Placement = App.Placement(obj.Placement.Base, combined_rotation)

# Re-compute the document
doc.recompute()
```