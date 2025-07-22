import React from 'react'
import CIcon from '@coreui/icons-react'
import {
  cilDescription,
  cilUser,
  cilStar,
  cilLockLocked,
  cilExitToApp,
  cilChatBubble,
  cilSpeedometer,
} from '@coreui/icons'
import { CNavGroup, CNavItem } from '@coreui/react'
const role = localStorage.getItem('role') // lấy role đã lưu

const _nav = [
  {
    component: CNavItem,
    name: 'Thống kê',
    to: '/dashboard',
    icon: <CIcon icon={cilSpeedometer} customClassName="nav-icon" />,
    badge: {
      color: 'info',
      text: 'NEW',
    },
  },
  {
    component: CNavItem,
    name: 'Quản lý tài liệu',
    to: '/quanly',
    icon: <CIcon icon={cilDescription} customClassName="nav-icon" />,
  },
  {
    component: CNavItem,
    name: 'Chatbot',
    to: '/chatbotpage',
    icon: <CIcon icon={cilChatBubble} customClassName="nav-icon" />,
  },
  {
    component: CNavItem,
    name: 'Quản lý người dùng',
    to: '/user',
    icon: <CIcon icon={cilUser} customClassName="nav-icon" />,
  },
  {
    component: CNavGroup,
    name: 'Quản lý người dùng',
    icon: <CIcon icon={cilStar} customClassName="nav-icon" />,
    items: [
      {
        component: CNavItem,
        name: 'Login',
        to: '/login',
        icon: <CIcon icon={cilLockLocked} customClassName="nav-icon" />,
      },
      {
        component: CNavItem,
        name: 'Register',
        to: '/register',
        icon: <CIcon icon={cilUser} customClassName="nav-icon" />,
      },
    ],
  },
  {
    component: CNavItem,
    name: 'Đăng xuất',
    to: '/logout',
    icon: <CIcon icon={cilExitToApp} customClassName="nav-icon" />,
  },
]

export default _nav
