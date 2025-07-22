import React, { useEffect, useState } from 'react'
import {
  CAvatar,
  CButton,
  CCard,
  CCardBody,
  CCardFooter,
  CCardHeader,
  CCol,
  CProgress,
  CRow,
  CTable,
  CTableBody,
  CTableDataCell,
  CTableHead,
  CTableHeaderCell,
  CTableRow,
  CFormSelect,
} from '@coreui/react'
import CIcon from '@coreui/icons-react'
import { cilCloudDownload, cilFile } from '@coreui/icons'
import axios from '../../api/axios'
import MainChart from './MainChart'

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalFiles: 0,
    deletedFiles: 0,
    fileTypes: {},
  })
  const [recentFiles, setRecentFiles] = useState([])

  const [dailyStats, setDailyStats] = useState([])
  const [weeklyStats, setWeeklyStats] = useState([])
  const [monthlyStats, setMonthlyStats] = useState([])

  const [chartMode, setChartMode] = useState('daily') // 'daily' | 'weekly' | 'monthly'

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await axios.get('/stats')
        setStats(res.data)
        setDailyStats(res.data.dailyStats || [])
        setWeeklyStats(res.data.weeklyStats || [])
        setMonthlyStats(res.data.monthlyStats || [])
      } catch (err) {
        console.error('Lỗi lấy thống kê:', err)
      }
    }

    const fetchRecentFiles = async () => {
      try {
        const res = await axios.get('/recent')
        setRecentFiles(res.data)
        console.log(res.data)
      } catch (err) {
        console.error('Lỗi lấy file gần đây:', err)
      }
    }

    fetchStats()
    fetchRecentFiles()
  }, [])

  const chartDataMap = {
    daily: { data: dailyStats, label: 'date', title: 'Biểu đồ theo ngày' },
    weekly: { data: weeklyStats, label: 'week', title: 'Biểu đồ theo tuần' },
    monthly: { data: monthlyStats, label: 'month', title: 'Biểu đồ theo tháng' },
  }

  const progressExample = [
    {
      title: 'Tài liệu đã tải lên',
      value: stats.totalFiles + ' file',
      percent: 100,
      color: 'success',
    },
    // {
    //   title: 'Tài liệu đã xóa',
    //   value: stats.deletedFiles,
    //   percent: stats.totalFiles > 0 ? Math.round((stats.deletedFiles / stats.totalFiles) * 100) : 0,
    //   color: 'danger',
    // },
    ...Object.entries(stats.types || {}).map(([type, count]) => ({
      title: `Loại ${type.toUpperCase()}`,
      value: count,
      percent: stats.totalFiles > 0 ? Math.round((count / stats.totalFiles) * 100) : 0,
      color: 'info',
    })),
  ]

  return (
    <>
      {/* Thống kê tổng quan */}
      <CCard className="mb-4">
        <CCardBody>
          <CRow>
            <CCol sm={5}>
              <h4 className="card-title mb-0">Thống kê tài liệu</h4>
              <div className="text-body-secondary">Hệ thống quản lý tài liệu</div>
            </CCol>
            <CCol sm={7} className="d-none d-md-block">
              <CButton color="primary" className="float-end">
                <CIcon icon={cilCloudDownload} />
              </CButton>
            </CCol>
          </CRow>
        </CCardBody>
        <CCardFooter>
          <CRow className="text-center">
            {progressExample.map((item, index) => (
              <CCol key={index}>
                <div className="text-body-secondary">{item.title}</div>
                <div className="fw-semibold text-truncate">
                  {item.value} ({item.percent}%)
                </div>
                <CProgress thin className="mt-2" color={item.color} value={item.percent} />
              </CCol>
            ))}
          </CRow>
        </CCardFooter>
      </CCard>

      {/* Biểu đồ chọn chế độ */}
      <CCard className="mb-4">
        <CCardHeader className="d-flex justify-content-between align-items-center">
          <span>{chartDataMap[chartMode].title}</span>
          <CFormSelect
            style={{ width: '200px' }}
            value={chartMode}
            onChange={(e) => setChartMode(e.target.value)}
          >
            <option value="daily">Theo ngày</option>
            <option value="weekly">Theo tuần</option>
            <option value="monthly">Theo tháng</option>
          </CFormSelect>
        </CCardHeader>
        <CCardBody>
          <MainChart data={chartDataMap[chartMode].data} labelKey={chartDataMap[chartMode].label} />
        </CCardBody>
      </CCard>

      {/* Danh sách tài liệu gần đây */}
      <CRow>
        <CCol xs>
          <CCard className="mb-4">
            <CCardHeader>Danh sách tài liệu gần đây</CCardHeader>
            <CCardBody>
              <CTable align="middle" className="mb-0 border" hover responsive>
                <CTableHead>
                  <CTableRow>
                    <CTableHeaderCell className="text-center">
                      <CIcon icon={cilFile} />
                    </CTableHeaderCell>
                    <CTableHeaderCell>Tên tài liệu</CTableHeaderCell>
                    <CTableHeaderCell>Loại</CTableHeaderCell>
                    <CTableHeaderCell>Người tải lên</CTableHeaderCell>
                    <CTableHeaderCell>Ngày tải lên</CTableHeaderCell>
                  </CTableRow>
                </CTableHead>
                <CTableBody>
                  {recentFiles.map((file, index) => (
                    <CTableRow key={index}>
                      <CTableDataCell className="text-center">
                        <CAvatar size="md" src="" />
                      </CTableDataCell>
                      <CTableDataCell>{file.name}</CTableDataCell>
                      <CTableDataCell>{file.type?.toUpperCase()}</CTableDataCell>
                      <CTableDataCell>{file.uploader || 'Không rõ'}</CTableDataCell>
                      <CTableDataCell>
                        {new Date(file.uploadDate || file.createdAt).toLocaleDateString()}
                      </CTableDataCell>
                    </CTableRow>
                  ))}
                </CTableBody>
              </CTable>
            </CCardBody>
          </CCard>
        </CCol>
      </CRow>
    </>
  )
}

export default Dashboard
