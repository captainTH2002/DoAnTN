import React from 'react'

const Colors = React.lazy(() => import('./views/theme/colors/Colors'))

const Quanly = React.lazy(() => import('./views/quanly/Quanly'))
const SearchSemantic = React.lazy(() => import('./views/search/SearchSemantic'))

const routes = [
  { path: '/', exact: true, name: 'Home' },
  { path: '/quanly', name: 'Quản lý tài liệu', element: Quanly },
  { path: '/search', name: 'Tìm kiếm', element: SearchSemantic },
  { path: '/theme/colors', name: 'Colors', element: Colors },
]

export default routes
