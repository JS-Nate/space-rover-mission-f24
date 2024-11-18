import React, { useState, useEffect } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import PlayerForm from "components/PlayerForm";
import GameScreen from "components/GameScreen";
import GameStateMessage from "components/GameStateMessage";
import BatteryStatus from "components/BatteryStatus";
import useGame, { GameState } from "hooks/useGame";
import { gameSocketURL, gameDurationSeconds } from "lib/config";

interface LocationState {
  playerName?: string;
}

// Tutorial timer hook moved outside component
const useTutorialTimer = (initialSeconds: number, isEnabled: boolean) => {
  const [seconds, setSeconds] = useState(initialSeconds);
  const navigate = useNavigate();

  useEffect(() => {
    if (!isEnabled) return;

    if (seconds <= 0) {
      navigate('/leaderboard?player=Tutorial&gameMode=1');
      return;
    }

    const timer = setInterval(() => {
      setSeconds(prev => prev - 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [seconds, navigate, isEnabled]);

  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  const formattedTime = `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;

  return formattedTime;
};

const PlayPage = () => {
  const location = useLocation();
  const state = location.state as LocationState;
  const isTutorialMode = state?.playerName === "Tutorial";

  // Both hooks are now called unconditionally
  const tutorialTime = useTutorialTimer(120, isTutorialMode);
  const {
    playerName,
    gameMode,
    gameModes,
    gameState,
    formattedTime,
    health,
    score,
    startGame,
    error,
    battery,
  } = useGame(gameSocketURL, gameDurationSeconds);

  // Early return for tutorial mode
  if (isTutorialMode) {
    return (
      <GameScreen
        playerName="Tutorial"
        gameMode="1"
        health={100}
        score={0}
        time={tutorialTime}
      />
    );
  }

  // Regular game flow
  switch (gameState) {
    case GameState.Connecting:
    case GameState.Error:
    case GameState.Waiting:
    case GameState.NotStarted:
      return (
        <div className="flex flex-col gap-7 justify-center h-full">
          <PlayerForm
            gameModes={gameModes}
            isDisabled={gameState !== GameState.NotStarted}
            onSubmit={startGame}
          />
          <GameStateMessage
            state={gameState}
            errorMessage={error}
          />
          <BatteryStatus
            batteryPercentage={battery}
          />
        </div>
      );

    case GameState.InGame:
      return (
        <GameScreen
          playerName={playerName}
          gameMode={gameMode}
          health={health}
          score={score}
          time={formattedTime}
        />
      );

    case GameState.GameEnded:
      return (
        <Navigate
          to={`/leaderboard?player=${encodeURIComponent(playerName)}&gameMode=${gameMode}`}
        />
      );

    default:
      return null;
  }
};

export default PlayPage;