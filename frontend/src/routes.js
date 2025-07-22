import React from 'react'

const Dashboard = React.lazy(() => import('./views/dashboard/Dashboard'))
const Quanly = React.lazy(() => import('./views/quanly/Quanly'))
const ChatbotPage = React.lazy(() => import('./views/chatbotpage/ChatbotPage'))
const User = React.lazy(() => import('./views/user/User'))

const routes = [
  { path: '/', exact: true, name: 'Home' },
  { path: '/quanly', name: 'Quản lý tài liệu', element: Quanly },
  { path: '/chatbotpage', name: 'Chatbot', element: ChatbotPage },
  { path: '/user', name: 'Quản lý người dùng', element: User },
  { path: '/dashboard', name: 'Thống kê', element: Dashboard },
]

export default routes
