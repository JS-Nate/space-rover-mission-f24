import React from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { QueryClientProvider } from "react-query";
import queryClient from "lib/react-query";
import Layout from "components/Layout";
import HomePage from "pages/HomePage";
import PlayPage from "pages/PlayPage";
import LeaderboardPage from "pages/LeaderboardPage";
import TutorialPage from "pages/Tutorial"

const App = () => {
  return (
    <QueryClientProvider client={queryClient as React.PropsWithChildren<typeof queryClient>}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/play" element={<PlayPage />} />
            <Route path="/tutorial" element={<TutorialPage />} />
            <Route path="/leaderboard" element={<LeaderboardPage />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  );
};

export default App;
