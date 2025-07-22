import React, { useState } from 'react'
import Chatbot from 'react-chatbot-kit'
import 'react-chatbot-kit/build/main.css'
import config from '../chatbotaction/config'
import MessageParser from '../chatbotaction/MessageParser'
import ActionProvider from '../chatbotaction/ActionProvider'
import './MyChatbot.css'

const MyChatbot = () => {
  const [showChat, setShowChat] = useState(false)

  return (
    <>
      {/* Nút nổi nhỏ kiểu Messenger */}
      <button
        onClick={() => setShowChat(!showChat)}
        style={{
          position: 'fixed',
          bottom: '30px',
          right: '30px',
          width: '60px',
          height: '60px',
          borderRadius: '50%',
          background: '#563d7c',
          color: 'white',
          fontSize: '24px',
          border: 'none',
          boxShadow: '0 4px 10px rgba(0,0,0,0.3)',
          cursor: 'pointer',
          zIndex: 9999,
        }}
      >
        💬
      </button>

      {/* Hộp Chatbot */}
      {showChat && (
        <div
          style={{
            position: 'fixed',
            bottom: '90px',
            right: '20px',
            width: '420px',
            height: '520px',
            boxShadow: '0 0 15px rgba(0,0,0,0.3)',
            borderRadius: '12px',
            overflow: 'hidden',
            zIndex: 9999,
            background: 'white',
          }}
        >
          <Chatbot config={config} messageParser={MessageParser} actionProvider={ActionProvider} />
        </div>
      )}
    </>
  )
}

export default MyChatbot

// import React, { useState, useEffect } from 'react'
// import Chatbot from 'react-chatbot-kit'
// import 'react-chatbot-kit/build/main.css'
// import config from '../chatbotaction/config'
// import MessageParser from '../chatbotaction/MessageParser'
// import ActionProvider from '../chatbotaction/ActionProvider'
// import axios from '../../api/axios'
// import './MyChatbot.css'

// const MyChatbot = () => {
//   const [showChat, setShowChat] = useState(false)
//   const [conversationId, setConversationId] = useState(null)

//   useEffect(() => {
//     if (showChat && !conversationId) {
//       axios.post('/conversations')
//         .then(res => setConversationId(res.data._id))
//         .catch(err => console.error('Không thể tạo conversation:', err))
//     }
//   }, [showChat, conversationId])

//   return (
//     <>
//       <button
//         onClick={() => setShowChat(!showChat)}
//         style={{
//           position: 'fixed',
//           bottom: '30px',
//           right: '30px',
//           width: '60px',
//           height: '60px',
//           borderRadius: '50%',
//           background: '#563d7c',
//           color: 'white',
//           fontSize: '24px',
//           border: 'none',
//           boxShadow: '0 4px 10px rgba(0,0,0,0.3)',
//           cursor: 'pointer',
//           zIndex: 9999,
//         }}
//       >
//         💬
//       </button>

//       {showChat && conversationId && (
//         <div
//           style={{
//             position: 'fixed',
//             bottom: '90px',
//             right: '20px',
//             width: '420px',
//             height: '520px',
//             boxShadow: '0 0 15px rgba(0,0,0,0.3)',
//             borderRadius: '12px',
//             overflow: 'hidden',
//             zIndex: 9999,
//             background: 'white',
//           }}
//         >
//           <Chatbot
//             config={config}
//             messageParser={(props) => new MessageParser({...props, conversationId})}
//             actionProvider={(props) => new ActionProvider({...props, conversationId})}
//           />
//         </div>
//       )}
//     </>
//   )
// }

// export default MyChatbot
