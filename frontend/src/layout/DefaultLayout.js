import React from 'react'
import { AppContent, AppSidebar, AppFooter, AppHeader } from '../components/index'
import MyChatbot from '../views/chatbot/MyChatbot'

const DefaultLayout = () => {
  return (
    <div>
      <AppSidebar />
      <div className="wrapper d-flex flex-column min-vh-100">
        <AppHeader />
        <div className="body flex-grow-1">
          <AppContent />
        </div>
        <AppFooter />
      </div>
      <MyChatbot />
    </div>
  )
}

export default DefaultLayout
