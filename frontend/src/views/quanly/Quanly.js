import React, { useState, useEffect } from 'react'
import {
  CTable,
  CButton,
  CModal,
  CModalHeader,
  CModalBody,
  CModalFooter,
  CFormInput,
  CFormLabel,
} from '@coreui/react'
import axios from '../../api/axios'
import { useAuth } from '../../contexts/AuthContext'

const Quanly = () => {
  const [visible, setVisible] = useState(false)
  const [editVisible, setEditVisible] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [items, setItems] = useState([])
  const [viewFile, setViewFile] = useState(null)
  const [viewImageUrl, setViewImageUrl] = useState(null)
  const [editFile, setEditFile] = useState({
    _id: '',
    name: '',
    documentType: '',
    tags: '',
  })
  const [page, setPage] = useState(1)
  const [limit] = useState(5)
  const [totalPages, setTotalPages] = useState(1)
  const [total, setTotal] = useState(0)
  const { user, loading } = useAuth()
  const [filters, setFilters] = useState({
    name: '',
    type: '',
    sizeMin: '',
    sizeMax: '',
    dateFrom: '',
    dateTo: '',
  })
  const handleClearFilters = () => {
    setFilters({
      name: '',
      type: '',
      sizeMin: '',
      sizeMax: '',
      dateFrom: '',
      dateTo: '',
    })
    setPage(1)
    loadDocuments()
  }
  const loadDocuments = async () => {
    try {
      const params = {
        page,
        limit,
        name: filters.name,
        sizeMin: filters.sizeMin,
        sizeMax: filters.sizeMax,
        dateFrom: filters.dateFrom,
        dateTo: filters.dateTo,
      }
      if (filters.type === 'image') {
        params.typeList = ['jpg', 'jpeg', 'png', 'webp', 'gif']
      } else if (filters.type) {
        params.type = filters.type
      }
      const query = new URLSearchParams(params).toString()
      const res = await axios.get(`/file?${query}`)
      const docs = res.data.data.map((doc, index) => ({
        id: (page - 1) * limit + index + 1,
        _id: doc._id,
        filename: doc.name,
        type: doc.type || doc.name.split('.').pop(),
        documentType: doc.documentType || 'Chưa xác định',
        tags: doc.tags || [],
        size: (doc.size / 1024).toFixed(2) + ' KB',
        uploadedAt: new Date(doc.createdAt).toLocaleDateString(),
      }))
      setItems(docs)
      setTotalPages(res.data.totalPages)
      setTotal(res.data.total)
    } catch (err) {
      console.error(err)
    }
  }

  useEffect(() => {
    if (user) loadDocuments()
  }, [filters, page, limit, user])

  const handleUpload = async () => {
    if (!selectedFile) return
    const formData = new FormData()
    formData.append('file', selectedFile)
    try {
      await axios.post('/file', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setVisible(false)
      setSelectedFile(null)
      alert('Tải lên thành công!')
      loadDocuments()
    } catch (err) {
      console.error(err)
      alert('Upload thất bại')
    }
  }

  const handleEdit = (item) => {
    setEditFile({
      _id: item._id,
      name: item.filename,
      documentType: item.documentType,
      tags: item.tags.join(', '),
    })
    setEditVisible(true)
  }

  const handleSaveEdit = async () => {
    try {
      await axios.put(`/file/${editFile._id}`, {
        name: editFile.name,
        documentType: editFile.documentType,
        tags: editFile.tags.split(',').map((tag) => tag.trim()),
      })
      alert('Cập nhật thành công!')
      setEditVisible(false)
      loadDocuments()
    } catch (err) {
      console.error(err)
      alert('Cập nhật thất bại')
    }
  }

  const handleDelete = async (item) => {
    if (user?.role !== 'admin') {
      alert('Bạn không có quyền thực hiện thao tác này')
      return
    }
    if (window.confirm(`Bạn có chắc muốn xóa ${item.filename}?`)) {
      try {
        await axios.delete(`/file/${item._id}`)
        alert('Đã xóa thành công!')
        loadDocuments()
      } catch (err) {
        console.error(err)
        alert(err.response?.data?.message || 'Xóa thất bại')
      }
    }
  }

  const handleView = async (item) => {
    setViewFile(item)
    console.log(item)
    // Nếu là ảnh thì tải về và tạo blob url
    if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'image/jpeg'].includes(item.type.toLowerCase())) {
      try {
        const res = await axios.get(`/file/${item._id}`, { responseType: 'blob' })
        setViewImageUrl(URL.createObjectURL(new Blob([res.data])))
      } catch (err) {
        console.error('Không tải được ảnh', err)
      }
    } else {
      setViewImageUrl(null)
    }
  }

  const handleDownload = async (file) => {
    try {
      console.log(file)
      const res = await axios.get(`/file/${file._id}`, { responseType: 'blob' })
      console.log(res.data)
      const url = window.URL.createObjectURL(new Blob([res.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', file.filename)
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    } catch (err) {
      console.error('Tải file thất bại', err)
    }
  }

  if (loading) return <div>Đang tải thông tin người dùng...</div>

  return (
    <>
      <h2 className="mb-4">Quản lý tài liệu</h2>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <div>
          Xin chào <strong>{user?.username}</strong>!
        </div>
      </div>

      <div className="mb-3">
        <div className="row g-2 align-items-end">
          <div className="col-md-2">
            <CFormInput
              placeholder="Tên tài liệu"
              value={filters.name}
              onChange={(e) => setFilters({ ...filters, name: e.target.value })}
            />
          </div>

          <div className="col-md-2">
            <select
              className="form-select"
              value={filters.type}
              onChange={(e) => setFilters({ ...filters, type: e.target.value })}
            >
              <option value="">Loại tài liệu</option>
              <option value="docx">Word</option>
              <option value="pdf">PDF</option>
              <option value="image">Ảnh</option>
            </select>
          </div>

          <div className="col-md-1">
            <CFormInput
              type="number"
              placeholder="Min KB"
              value={filters.sizeMin}
              onChange={(e) => setFilters({ ...filters, sizeMin: e.target.value })}
            />
          </div>

          <div className="col-md-1">
            <CFormInput
              type="number"
              placeholder="Max KB"
              value={filters.sizeMax}
              onChange={(e) => setFilters({ ...filters, sizeMax: e.target.value })}
            />
          </div>

          <div className="col-md-2">
            <CFormInput
              type="date"
              placeholder="Từ ngày"
              value={filters.dateFrom}
              onChange={(e) => setFilters({ ...filters, dateFrom: e.target.value })}
            />
          </div>

          <div className="col-md-2">
            <CFormInput
              type="date"
              placeholder="Đến ngày"
              value={filters.dateTo}
              min={filters.dateFrom} // 👈 Giới hạn không cho nhỏ hơn "Từ ngày"
              onChange={(e) => setFilters({ ...filters, dateTo: e.target.value })}
            />
          </div>

          <div className="col-md-2 d-flex gap-1 justify-content-end">
            <CButton color="secondary" onClick={handleClearFilters}>
              Xóa nhanh
            </CButton>
            <CButton color="primary" onClick={() => setVisible(true)}>
              Tải tài liệu
            </CButton>
          </div>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <CTable bordered>
          <thead>
            <tr>
              <th className="text-center align-middle">#</th>
              <th className="text-center align-middle">Tên tài liệu</th>
              <th className="text-center align-middle">Loại tài liệu</th>
              <th className="text-center align-middle">Kích thước</th>
              <th className="text-center align-middle">Ngày tải lên</th>
              <th className="text-center align-middle">Thao tác</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.id}>
                <td className="text-center align-middle">{item.id}</td>
                <td className="align-middle">{item.filename}</td>
                <td className="text-center align-middle">{item.type}</td>
                <td className="text-center align-middle">{item.size}</td>
                <td className="text-center align-middle">{item.uploadedAt}</td>
                <td className="text-center align-middle" style={{ width: '220px' }}>
                  <CButton color="info" size="sm" className="me-1" onClick={() => handleView(item)}>
                    Xem
                  </CButton>
                  <CButton
                    color="warning"
                    size="sm"
                    className="me-1"
                    onClick={() => handleEdit(item)}
                  >
                    Sửa
                  </CButton>
                  <CButton color="danger" size="sm" onClick={() => handleDelete(item)}>
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

      {/* Modal Upload */}
      <CModal visible={visible} onClose={() => setVisible(false)}>
        <CModalHeader>Tải tài liệu</CModalHeader>
        <CModalBody>
          <input type="file" onChange={(e) => setSelectedFile(e.target.files[0])} />
          <p className="mt-2 text-muted">Chọn file bất kỳ (ảnh hoặc tài liệu)</p>
        </CModalBody>
        <CModalFooter>
          <CButton color="secondary" onClick={() => setVisible(false)}>
            Đóng
          </CButton>
          <CButton color="primary" onClick={handleUpload} disabled={!selectedFile}>
            Upload
          </CButton>
        </CModalFooter>
      </CModal>

      {/* Modal Xem chi tiết */}
      <CModal visible={!!viewFile} onClose={() => setViewFile(null)}>
        <CModalHeader>Chi tiết tài liệu</CModalHeader>
        <CModalBody>
          {viewFile && (
            <>
              <p>
                <strong>Tên:</strong> {viewFile.filename}
              </p>
              <p>
                <strong>Loại:</strong> {viewFile.type}
              </p>
              <p>
                <strong>Loại văn bản:</strong> {viewFile.documentType}
              </p>
              <p>
                <strong>Kích thước:</strong> {viewFile.size}
              </p>
              <p>
                <strong>Ngày tải:</strong> {viewFile.uploadedAt}
              </p>
              <p>
                <strong>Tags:</strong> {viewFile.tags.join(', ')}
              </p>
              {viewImageUrl ? (
                <div className="mt-3 text-center">
                  <img
                    src={viewImageUrl}
                    alt={viewFile.filename}
                    style={{ maxWidth: '100%', maxHeight: '400px' }}
                  />
                </div>
              ) : (
                <div className="mt-3">
                  <CButton color="primary" onClick={() => handleDownload(viewFile)}>
                    Tải file về
                  </CButton>
                </div>
              )}
            </>
          )}
        </CModalBody>
        <CModalFooter>
          <CButton color="secondary" onClick={() => setViewFile(null)}>
            Đóng
          </CButton>
        </CModalFooter>
      </CModal>

      {/* Modal Sửa */}
      <CModal visible={editVisible} onClose={() => setEditVisible(false)}>
        <CModalHeader>Sửa tài liệu</CModalHeader>
        <CModalBody>
          <div className="mb-3">
            <CFormLabel>Tên tài liệu</CFormLabel>
            <CFormInput
              value={editFile.name}
              onChange={(e) => setEditFile({ ...editFile, name: e.target.value })}
            />
          </div>
          <div className="mb-3">
            <CFormLabel>Loại văn bản</CFormLabel>
            <CFormInput
              value={editFile.documentType}
              onChange={(e) => setEditFile({ ...editFile, documentType: e.target.value })}
            />
          </div>
          <div className="mb-3">
            <CFormLabel>Tags (phân tách dấu phẩy)</CFormLabel>
            <CFormInput
              value={editFile.tags}
              onChange={(e) => setEditFile({ ...editFile, tags: e.target.value })}
            />
          </div>
        </CModalBody>
        <CModalFooter>
          <CButton color="secondary" onClick={() => setEditVisible(false)}>
            Đóng
          </CButton>
          <CButton color="primary" onClick={handleSaveEdit}>
            Lưu
          </CButton>
        </CModalFooter>
      </CModal>
    </>
  )
}

export default Quanly
