
__device__ __forceinline__ void Get_Normalized(float3 &ray_direction)
{
    float m_x = ray_direction.x;
    float m_y = ray_direction.y;
    float m_z = ray_direction.z;
    float temp = 1 / sqrtf(m_x * m_x + m_y * m_y + m_z * m_z);
    ray_direction.x = m_x * temp;
    ray_direction.y = m_y * temp;
    ray_direction.z = m_z * temp;
}