// import React from 'react';

// export default function MessageBubble({ sender, text }) {
//   const isUser = sender === 'user';

//   return (
//     <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'}`}>
//       <div
//         className={`max-w-[75%] px-4 py-2 text-sm leading-relaxed shadow-md transition-all duration-200
//           ${isUser
//             ? 'bg-blue-600 text-white rounded-tl-2xl rounded-bl-2xl rounded-br-md'
//             : 'bg-gray-200 text-gray-900 rounded-tr-2xl rounded-br-2xl rounded-bl-md'
//           }`}
//       >
//         {text}
//       </div>
//     </div>
//   );
// }
import React from "react";

export default function MessageBubble({ sender, text }) {
  const isUser = sender === "user";

  return (
    <div className={`flex w-full ${isUser ? "justify-start" : "justify-end"}`}>
      <div
        className={`max-w-[70%] px-4 py-2 text-sm leading-relaxed break-words transition-all duration-200 transform hover:scale-105 ${
          isUser
            ? "bg-blue-100 text-gray-900 rounded-tl-2xl rounded-tr-2xl rounded-br-md border border-blue-200 shadow-md"
            : "bg-blue-600 text-white rounded-tl-2xl rounded-tr-2xl rounded-bl-md shadow-lg"
        }`}
      >
        {text}
      </div>
    </div>
  );
}






// import React from 'react';

// export default function MessageBubble({ sender, text }) {
//   const isUser = sender === 'user';

//   return (
//     <div
//       className={`flex w-full ${
//         isUser ? 'justify-start' : 'justify-end'
//       }`}
//     >
//       <div
//         className={`max-w-[75%] px-4 py-2 text-sm leading-relaxed shadow-md text-white 
//           ${
//             isUser
//               ? 'bg-blue-600 rounded-tr-2xl rounded-bl-2xl rounded-br-md'
//               : 'bg-gray-500 rounded-tl-2xl rounded-br-2xl rounded-bl-md'
//           }`}
//       >
//         {text}
//       </div>
//     </div>
//   );
// }
