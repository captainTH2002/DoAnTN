import React, { useEffect, useState } from 'react'
import axios from '../../api/axios'
import {
  CTable,
  CButton,
  CModal,
  CModalHeader,
  CModalBody,
  CModalFooter,
  CFormInput,
  CFormLabel,
  CFormSelect,
} from '@coreui/react'
import { useAuth } from '../../contexts/AuthContext'

const QuanlyUser = () => {
  const [users, setUsers] = useState([])
  const [editVisible, setEditVisible] = useState(false)
  const [editUser, setEditUser] = useState({ _id: '', username: '', role: '', status: '' })
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [total, setTotal] = useState(0)
  const limit = 5
  const { user, refreshUser } = useAuth()
  const loadUsers = async () => {
    try {
      const res = await axios.get(`/users?page=${page}&limit=${limit}`)
      setUsers(res.data.data)
      setTotalPages(res.data.totalPages)
      setTotal(res.data.total)
    } catch (err) {
      console.error('Lỗi loadUsers:', err)
      alert('Không tải được danh sách người dùng')
    }
  }

  useEffect(() => {
    loadUsers()
  }, [page])

  const handleEdit = (user) => {
    setEditUser({ _id: user._id, username: user.username, role: user.role, status: user.status })
    setEditVisible(true)
  }

  const handleSave = async () => {
    try {
      await axios.put(`/users/${editUser._id}`, {
        username: editUser.username,
        role: editUser.role,
        status: editUser.status,
      })
      alert('Cập nhật thành công')
      setEditVisible(false)
      refreshUser()
      loadUsers()
    } catch (err) {
      console.error('Lỗi cập nhật user:', err)
      alert('Cập nhật thất bại')
    }
  }

  const handleDelete = async (id) => {
    if (!window.confirm('Xác nhận xóa user?')) return
    try {
      await axios.delete(`/users/${id}`)
      alert('Đã xóa')
      loadUsers()
    } catch (err) {
      console.error('Lỗi xóa user:', err)
      alert('Xóa thất bại')
    }
  }

  const toggleStatus = async (id, currentStatus) => {
    const newStatus = currentStatus === 'active' ? 'inactive' : 'active'
    try {
      await axios.put(`/users/${id}`, { status: newStatus })
      loadUsers()
    } catch (err) {
      console.error('Lỗi đổi trạng thái:', err)
      alert('Không đổi được trạng thái')
    }
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: '80vh',
      }}
    >
      <h5 className="mb-4">Quản lý người dùng</h5>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <div>
          Xin chào <strong>{user?.username}</strong>
        </div>
      </div>
      <div style={{ width: '90%', margin: '0 auto', overflowX: 'auto' }}>
        <CTable bordered responsive>
          <thead>
            <tr className="text-center align-middle">
              <th>#</th>
              <th>Username</th>
              <th>Email</th>
              <th>Quyền</th>
              <th>Trạng thái</th>
              <th>Hành động</th>
            </tr>
          </thead>
          <tbody>
            {users.map((user, index) => (
              <tr key={user._id} className="text-center align-middle">
                <td>{(page - 1) * limit + index + 1}</td>
                <td>{user.username}</td>
                <td>{user.email}</td>
                <td>{user.role}</td>
                <td>
                  <input
                    type="checkbox"
                    checked={user.status === 'active'}
                    onChange={() => toggleStatus(user._id, user.status)}
                    style={{ transform: 'scale(1.4)', cursor: 'pointer' }}
                  />
                </td>
                <td>
                  <CButton
                    color="warning"
                    size="sm"
                    className="me-2"
                    onClick={() => handleEdit(user)}
                  >
                    Sửa
                  </CButton>
                  <CButton color="danger" size="sm" onClick={() => handleDelete(user._id)}>
                    Xóa
                  </CButton>
                </td>
              </tr>
            ))}
          </tbody>
        </CTable>
        <div
          className="d-flex justify-content-between align-items-center my-3 px-2"
          style={{ borderTop: '1px solid rgba(255,255,255,0.2)', paddingTop: '0.5rem' }}
        >
          <div>
            Tổng: <strong>{total}</strong> mục | Trang <strong>{page}</strong> /{' '}
            <strong>{totalPages}</strong>
          </div>
          <div>
            <nav>
              <ul className="pagination mb-0">
                <li className={`page-item ${page === 1 ? 'disabled' : ''}`}>
                  <button className="page-link" onClick={() => setPage(1)}>
                    &laquo;
                  </button>
                </li>
                <li className={`page-item ${page === 1 ? 'disabled' : ''}`}>
                  <button className="page-link" onClick={() => setPage(page - 1)}>
                    &lsaquo;
                  </button>
                </li>

                {page > 2 && (
                  <li className="page-item">
                    <span className="page-link">...</span>
                  </li>
                )}
                {page > 1 && (
                  <li className="page-item">
                    <button className="page-link" onClick={() => setPage(page - 1)}>
                      {page - 1}
                    </button>
                  </li>
                )}
                <li className="page-item active">
                  <span className="page-link">{page}</span>
                </li>
                {page < totalPages && (
                  <li className="page-item">
                    <button className="page-link" onClick={() => setPage(page + 1)}>
                      {page + 1}
                    </button>
                  </li>
                )}
                {page < totalPages - 1 && (
                  <li className="page-item">
                    <span className="page-link">...</span>
                  </li>
                )}

                <li className={`page-item ${page === totalPages ? 'disabled' : ''}`}>
                  <button className="page-link" onClick={() => setPage(page + 1)}>
                    &rsaquo;
                  </button>
                </li>
                <li className={`page-item ${page === totalPages ? 'disabled' : ''}`}>
                  <button className="page-link" onClick={() => setPage(totalPages)}>
                    &raquo;
                  </button>
                </li>
              </ul>
            </nav>
          </div>
        </div>
      </div>

      {/* Modal sửa */}
      <CModal visible={editVisible} onClose={() => setEditVisible(false)}>
        <CModalHeader>Sửa người dùng</CModalHeader>
        <CModalBody>
          <div className="mb-3">
            <CFormLabel>Username</CFormLabel>
            <CFormInput
              value={editUser.username}
              onChange={(e) => setEditUser({ ...editUser, username: e.target.value })}
            />
          </div>
          <div className="mb-3">
            <CFormLabel>Quyền</CFormLabel>
            <CFormSelect
              value={editUser.role}
              onChange={(e) => setEditUser({ ...editUser, role: e.target.value })}
            >
              <option value="">Chọn quyền</option>
              <option value="user">User</option>
              <option value="admin">Admin</option>
            </CFormSelect>
          </div>
          <div className="mb-3">
            <CFormLabel>Trạng thái</CFormLabel>
            <CFormSelect
              value={editUser.status}
              onChange={(e) => setEditUser({ ...editUser, status: e.target.value })}
            >
              <option value="">Chọn trạng thái</option>
              <option value="active">Hoạt động</option>
              <option value="inactive">Khóa</option>
            </CFormSelect>
          </div>
        </CModalBody>
        <CModalFooter>
          <CButton color="secondary" onClick={() => setEditVisible(false)}>
            Hủy
          </CButton>
          <CButton color="primary" onClick={handleSave}>
            Lưu
          </CButton>
        </CModalFooter>
      </CModal>
    </div>
  )
}

export default QuanlyUser
