// import React, { useState, useEffect, useRef } from 'react';
// import axios from 'axios';
// import MessageBubble from '../components/MessageBubble';
// import Loader from '../components/Loader';
// import { Send } from 'lucide-react';

// export default function ChatPage() {
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState('');
//   const [loading, setLoading] = useState(false);
//   const scrollRef = useRef(null);

//   const API_BASE = 'http://127.0.0.1:8000';

//   const getSessionId = () => {
//     let sessionId = localStorage.getItem('session_id');
//     if (!sessionId) {
//       sessionId = 'sess_' + Math.random().toString(36).substring(2, 10);
//       localStorage.setItem('session_id', sessionId);
//     }
//     return sessionId;
//   };

//   const startSession = async () => {
//     const sessionId = getSessionId();
//     try {
//       const res = await axios.post(`${API_BASE}/sessions`, { user_id: sessionId });
//       console.log("✅ Session started:", res.data.session_id);
//     } catch (err) {
//       console.error('❌ Failed to start session:', err.response?.data || err.message);
//     }
//   };

//   // autoscroll when messages change
//   useEffect(() => {
//     if (scrollRef.current) {
//       scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
//     }
//   }, [messages]);

//   useEffect(() => {
//     startSession();
//   }, []);

//   const sendMessage = async () => {
//     if (!input.trim()) return;
//     const userMsg = { sender: 'user', text: input };
//     setMessages((prev) => [...prev, userMsg]);
//     setInput('');
//     setLoading(true);

//     const sessionId = getSessionId();

//     try {
//       const res = await axios.post(`${API_BASE}/sessions/${sessionId}/message`, { message: userMsg.text });
//       const botMsg = { sender: 'bot', text: res.data.reply || 'No response.' };
//       setMessages((prev) => [...prev, botMsg]);
//     } catch (err) {
//       console.error('❌ Server error:', err.response?.data || err.message);
//       setMessages((prev) => [...prev, { sender: 'bot', text: '⚠️ Server error. Please try again later.' }]);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const handleKeyDown = (e) => {
//     if (e.key === 'Enter' && !e.shiftKey) {
//       e.preventDefault();
//       sendMessage();
//     }
//   };

//   return (
//     <div className="min-h-screen flex items-center justify-center p-6 bg-[radial-gradient(ellipse_at_top_left,_var(--tw-gradient-stops))] from-sky-50 to-white">
//       <div className="flex flex-col items-center">
//         <h1 className="text-2xl sm:text-3xl font-semibold text-sky-700 mb-6">AI Customer Support</h1>

//         {/* Outer decorative frame with gradient border */}
//         <div className="relative">
//           <div
//             className="rounded-3xl p-[2px] bg-gradient-to-r from-indigo-400 via-sky-400 to-emerald-300 shadow-2xl"
//             style={{ width: 420 }}
//           >
//             {/* inner panel */}
//             <div className="rounded-3xl bg-white/80 backdrop-blur-sm border border-white/40" style={{ width: 420 }}>
//               {/* Chat title strip */}
//               <div className="px-5 py-3 border-b border-white/30 rounded-t-3xl bg-gradient-to-b from-white/60 to-transparent">
//                 <div className="flex items-center justify-between">
//                   <div className="text-sm text-gray-700 font-medium">Support Assistant</div>
//                   <div className="text-xs text-gray-500">online</div>
//                 </div>
//               </div>

//               {/* Chat area: fixed height, scrollable */}
//               <div
//                 ref={scrollRef}
//                 className="px-4 py-4 h-[560px] overflow-y-auto space-y-4"
//                 style={{ width: 420 }}
//               >
//                 {messages.length === 0 && (
//                   <p className="text-gray-400 text-center mt-8 italic">Start chatting with our AI assistant 💬</p>
//                 )}
//                 {messages.map((m, idx) => (
//                   <MessageBubble key={idx} sender={m.sender} text={m.text} />
//                 ))}
//               </div>

//               {/* Input area */}
//               <div className="px-4 py-4 border-t border-white/30 rounded-b-3xl bg-white/70 flex items-center gap-3">
//                 <textarea
//                   value={input}
//                   onChange={(e) => setInput(e.target.value)}
//                   onKeyDown={handleKeyDown}
//                   placeholder="Type your message... (Press Enter to send)"
//                   className="flex-1 resize-none px-3 py-2 rounded-xl border border-gray-200 focus:outline-none focus:ring-2 focus:ring-sky-300 h-12 bg-white text-sm"
//                 />
//                 <button
//                   onClick={sendMessage}
//                   disabled={loading}
//                   className="bg-sky-600 hover:bg-sky-700 text-white px-4 py-2 rounded-xl flex items-center gap-2 shadow"
//                 >
//                   {loading ? <Loader /> : <Send size={16} />}
//                   <span className="hidden sm:inline text-sm">Send</span>
//                 </button>
//               </div>
//             </div>
//           </div>

//           {/* small decorative glow */}
//           <div className="absolute -bottom-4 left-1/2 -translate-x-1/2 w-[260px] h-6 rounded-full blur-2xl opacity-20 bg-gradient-to-r from-indigo-300 to-emerald-200" />
//         </div>
//       </div>
//     </div>
//   );
// }

import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import MessageBubble from "../components/MessageBubble";
import Loader from "../components/Loader";
import { Send } from "lucide-react";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef(null);
  const API_BASE = "http://127.0.0.1:8000";

  // ✅ Session Handling
  const getSessionId = () => {
    let sessionId = localStorage.getItem("session_id");
    if (!sessionId) {
      sessionId = "sess_" + Math.random().toString(36).substring(2, 10);
      localStorage.setItem("session_id", sessionId);
    }
    return sessionId;
  };

  const startSession = async () => {
    const sessionId = getSessionId();
    try {
      await axios.post(`${API_BASE}/sessions`, { user_id: sessionId });
    } catch (err) {
      console.error("Failed to start session:", err.message);
    }
  };

  useEffect(() => {
    startSession();
  }, []);

  useEffect(() => {
    if (scrollRef.current)
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  // ✅ Send message logic
  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMsg = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setIsTyping(true);
    const sessionId = getSessionId();

    try {
      const res = await axios.post(`${API_BASE}/sessions/${sessionId}/message`, {
        message: userMsg.text,
      });
      const botMsg = { sender: "bot", text: res.data.reply || "No response." };
      setTimeout(() => {
        setMessages((prev) => [...prev, botMsg]);
        setIsTyping(false);
      }, 600);
    } catch {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Server error. Try again later." },
      ]);
      setIsTyping(false);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-sky-50 via-indigo-50 to-white p-6">
      {/* Chat Frame */}
      <div className="relative w-[480px] h-[680px] backdrop-blur-sm bg-white/80 rounded-3xl shadow-2xl border border-gray-200 flex flex-col overflow-hidden">
        
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-blue-500 text-white px-6 py-4 shadow-md text-center">
          <h1 className="text-xl font-semibold">AI Support Assistant</h1>
          <p className="text-sm text-blue-200 mt-1">Online</p>
        </div>

        {/* Chat Area */}
        <div
          ref={scrollRef}
          className="flex flex-col gap-3 px-5 py-4 flex-1 overflow-y-auto"
        >
          {messages.length === 0 && (
            <p className="text-gray-400 text-center mt-20 italic">
              Start chatting with our AI assistant!
            </p>
          )}
          {messages.map((m, i) => (
            <MessageBubble key={i} sender={m.sender} text={m.text} />
          ))}
          {isTyping && (
            <div className="flex justify-end">
              <div className="bg-blue-600 text-white px-4 py-2 rounded-tl-2xl rounded-tr-2xl rounded-bl-md animate-pulse shadow-lg text-sm">
                Typing...
              </div>
            </div>
          )}
        </div>

        {/* Input + Send Section */}
        <div className="px-4 py-3 border-t border-gray-200 bg-white flex items-center gap-3">
          {/* Text Area */}
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            rows={2}
            className="flex-1 resize-none rounded-2xl border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 bg-gray-50"
          />
          {/* Send Button (Right-Aligned) */}
          <button
            onClick={sendMessage}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-md flex items-center justify-center transition-transform hover:scale-105 disabled:opacity-60"
          >
            {loading ? <Loader /> : <Send size={18} />}
          </button>
        </div>
      </div>
    </div>
  );
}
