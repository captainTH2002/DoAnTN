// src/components/RoleRoute.jsx
import React from 'react'
import { useAuth } from '../contexts/AuthContext'

const RoleRoute = ({ children, roles }) => {
  const { user, loading, isAuthenticated } = useAuth()

  if (loading) {
    return <div style={styles.loadingBox}>Đang kiểm tra quyền...</div>
  }

  if (!isAuthenticated) {
    return (
      <div style={styles.errorBox}>
        <h2>🚪 Bạn chưa đăng nhập</h2>
        <p>Vui lòng đăng nhập để tiếp tục sử dụng hệ thống.</p>
      </div>
    )
  }

  if (!roles.includes(user.role)) {
    return (
      <div style={styles.errorBox}>
        <h2>🚫 Không đủ quyền truy cập</h2>
        <p>
          Tài khoản của bạn (<strong>{user.role}</strong>) không có quyền xem trang này.
        </p>
      </div>
    )
  }

  return children
}

const styles = {
  errorBox: {
    maxWidth: '500px',
    margin: '50px auto',
    padding: '30px',
    border: '2px solid #f44336',
    borderRadius: '10px',
    textAlign: 'center',
    color: '#f44336',
    background: '#fff5f5',
  },
  loadingBox: {
    maxWidth: '300px',
    margin: '50px auto',
    padding: '20px',
    border: '2px dashed #3f51b5',
    borderRadius: '10px',
    textAlign: 'center',
    color: '#3f51b5',
    background: '#f0f7ff',
  }
}

export default RoleRoute
