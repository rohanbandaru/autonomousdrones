import numpy as np
import numpy.linalg as npl
from typing import Union, Tuple


class Orientation:
    """
    A class representing 3D orientation using quaternions.
    Quaternions are stored in [w, x, y, z] format where w is the scalar component.
    """
    def __init__(self, quat: Union[np.ndarray, list] = None):
        """
        Initialize an Orientation.
        
        Args:
            quat: Optional quaternion as [w, x, y, z]. Defaults to identity rotation.
        """
        if quat is None:
            self._quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            self._quat = np.array(quat)
            self._normalize()
    
    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> 'Orientation':
        """
        Create orientation from axis-angle representation.
        
        Args:
            axis: Unit vector representing rotation axis
            angle: Rotation angle in radians
        """
        axis = np.array(axis)
        axis = axis / npl.norm(axis)
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        
        quat = np.array([
            np.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        ])
        
        return Orientation(quat)
    
    @staticmethod
    def from_euler(angles: Union[np.ndarray, list], order: str = 'xyz') -> 'Orientation':
        """
        Create orientation from Euler angles.
        
        Args:
            angles: Rotation angles in radians [x, y, z]
            order: Rotation order, default 'xyz'
        """
        angles = np.array(angles)
        quats = []
        
        for axis, angle in zip(order, angles):
            if axis == 'x':
                axis_vec = [1, 0, 0]
            elif axis == 'y':
                axis_vec = [0, 1, 0]
            else:  # z
                axis_vec = [0, 0, 1]
            
            quats.append(Orientation.from_axis_angle(axis_vec, angle))
        
        # Compose rotations
        result = quats[0]
        for q in quats[1:]:
            result = result.compose(q)
        
        return result
    
    def _normalize(self) -> None:
        """Normalize the internal quaternion to prevent numerical drift."""
        self._quat = self._quat / npl.norm(self._quat)
    
    @property
    def quaternion(self) -> np.ndarray:
        """Get the internal quaternion."""
        return self._quat.copy()
    
    def compose(self, other: 'Orientation') -> 'Orientation':
        """
        Compose this rotation with another rotation (this * other).
        Result is equivalent to applying other rotation first, then this rotation.
        
        Args:
            other: Another Orientation to compose with
        """
        return Orientation(self._quat_multiply(self._quat, other._quat))
    
    def inverse(self) -> 'Orientation':
        """Return the inverse rotation."""
        # For unit quaternions, inverse is the conjugate
        return Orientation([self._quat[0], -self._quat[1], -self._quat[2], -self._quat[3]])
    
    def rotate(self, vector: Union[np.ndarray, list]) -> np.ndarray:
        """
        Rotate a vector by this orientation.
        
        Args:
            vector: 3D vector to rotate
            
        Returns:
            Rotated vector
        """
        v = np.array(vector)
        # Convert vector to quaternion with w=0
        v_quat = np.array([0.0, v[0], v[1], v[2]])
        
        # Apply rotation: q * v * q'
        q_conj = self.inverse().quaternion
        result = self._quat_multiply(
            self._quat_multiply(self._quat, v_quat),
            q_conj
        )
        
        # Return vector part
        return result[1:]
    
    @staticmethod
    def between_vectors(v1: np.ndarray, v2: np.ndarray) -> 'Orientation':
        # Normalize the input vectors
        v1 = v1 / npl.norm(v1)
        v2 = v2 / npl.norm(v2)
        
        # Compute the axis of rotation (cross product)
        axis = np.cross(v2, v1)
        axis_norm = npl.norm(axis)
        
        if axis_norm < 1e-6:  # Vectors are parallel or anti-parallel
            if np.dot(v2, v1) > 0:  # Parallel case (no rotation needed)
                return np.array([0, 0, 0, 1])
            else:  # Anti-parallel case (180-degree rotation around any perpendicular axis)
                return np.array([1, 0, 0, 0])  # Rotate around the x-axis
        
        # Normalize the axis of rotation
        axis = axis / axis_norm
        
        # Compute the angle of rotation (dot product and arccos)
        angle = np.arccos(np.dot(v2, v1))
        
        # Construct the quaternion (axis-angle to quaternion conversion)
        return Orientation.from_axis_angle(axis, angle)
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 rotation matrix."""
        w, x, y, z = self._quat
        
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
            [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def to_axis_angle(self) -> Tuple[np.ndarray, float]:
        """
        Convert quaternion to axis-angle representation.
        
        Returns:
            Tuple of (axis, angle) where axis is a unit vector and angle is in radians.
        """
        w, x, y, z = self._quat
        angle = 2 * np.arccos(w)
        sin_half = np.sin(angle / 2.0)
        if np.abs(angle) < 1e-6: # No rotation
            return np.array([1, 0, 0]), 0.0
        return np.array([x, y, z]) / sin_half, angle
    
    def to_euler(self, order: str = 'xyz') -> np.ndarray:
        """
        Convert to Euler angles in specified order.
        
        Args:
            order: Rotation order, default 'xyz'
            
        Returns:
            Euler angles in radians [x, y, z]
        """
        R = self.to_matrix()
        
        if order == 'xyz':
            y = np.arcsin(-R[2, 0])
            if np.abs(R[2, 0]) < 0.999999:
                x = np.arctan2(R[2, 1], R[2, 2])
                z = np.arctan2(R[1, 0], R[0, 0])
            else:  # Gimbal lock
                x = np.arctan2(-R[1, 2], R[1, 1])
                z = 0
        else:
            raise NotImplementedError(f"Euler order {order} not implemented")
            
        return np.array([x, y, z])
    
    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Helper method for quaternion multiplication."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def __mul__(self, other: 'Orientation') -> 'Orientation':
        """Operator overloading for *."""
        return self.compose(other)

# Example usage:
if __name__ == "__main__":
    # Create rotation of 90 degrees around x-axis
    rot_x = Orientation.from_axis_angle([1, 0, 0], np.pi/2)
    
    # Create rotation of 45 degrees around z-axis
    rot_z = Orientation.from_axis_angle([0, 0, 1], np.pi/4)
    
    # Compose rotations
    combined = rot_x * rot_z
    
    # Rotate a vector
    vec = np.array([1, 0, 0])
    rotated = combined.rotate(vec)
    
    print(f"Original vector: {vec}")
    print(f"Rotated vector: {rotated}")
    
    # Convert to Euler angles
    euler = combined.to_euler()
    print(f"Euler angles (rad): {euler}")
    print(f"Euler angles (deg): {np.degrees(euler)}")

    rot_test = Orientation.from_euler([1.57079, -0.78539, 0])
    print(f"Euler angles (rad): {rot_test.to_euler()}")