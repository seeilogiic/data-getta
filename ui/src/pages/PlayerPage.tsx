import { useState, useEffect } from 'react';
import { useParams, useLocation, useNavigate, Outlet, useSearchParams } from 'react-router';
import { supabase } from '@/utils/supabase/client';
import { PlayersTable } from '@/types/schemas';
import Box from '@mui/material/Box';
import ModelTabs from '@/components/player/ModelTabs';
import PlayerInfo from '@/components/player/PlayerInfo';

export default function PlayerPage() {
  const { trackmanAbbreviation, playerName } = useParams<{
    trackmanAbbreviation: string;
    playerName: string;
  }>();
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const roleParam = searchParams.get('role');

  // Narrow role to only 'batter' | 'pitcher', default to 'batter' if invalid or missing
  const role: 'batter' | 'pitcher' = roleParam === 'pitcher' ? 'pitcher' : 'batter';

  const [player, setPlayer] = useState<PlayersTable | null>(null);
  const [loading, setLoading] = useState(true);

  // Redirect base player URL to stats for current year
  useEffect(() => {
    if (trackmanAbbreviation && playerName) {
      const playerPath = `/team/${trackmanAbbreviation}/player/${playerName}`;
      if (location.pathname === playerPath) {
        navigate(`${playerPath}/stats/2025?role=${role}`, { replace: true });
      }
    }
  }, [location.pathname, trackmanAbbreviation, playerName, navigate, role]);

  // Fetch player info
  useEffect(() => {
    async function fetchPlayer() {
      if (!trackmanAbbreviation || !playerName) return;

      try {
        const decodedTrackmanAbbreviation = decodeURIComponent(trackmanAbbreviation);
        const decodedPlayerName = decodeURIComponent(playerName).replace('_', ', ');
        setLoading(true);

        const { data, error } = await supabase
          .from('Players')
          .select('*')
          .eq('TeamTrackmanAbbreviation', decodedTrackmanAbbreviation)
          .eq('Name', decodedPlayerName)
          .eq('Year', 2025)
          .maybeSingle();

        if (error) throw error;
        setPlayer(data || null);
      } catch (error) {
        console.error('Error fetching player:', error);
      } finally {
        setLoading(false);
      }
    }

    fetchPlayer();
  }, [trackmanAbbreviation, playerName]);

  if (loading) return <div>Loading...</div>;
  if (!player) return <div>Player not found</div>;

  const decodedTeamName = decodeURIComponent(trackmanAbbreviation || '');
  const decodedPlayerName = decodeURIComponent(playerName || '');

  return (
    <Box>
      {/* Tabs */}
      <Box
        sx={{
          backgroundColor: '#f5f5f5',
          paddingLeft: { xs: 4, sm: 8 },
          paddingY: 2,
          marginTop: '4px',
        }}
      >
        {/* Pass narrowed role to ModelTabs */}
        <ModelTabs
          team={decodedTeamName}
          player={decodedPlayerName}
          role={role}
        />
      </Box>

      {/* Player info and nested content */}
      <Box sx={{ paddingX: { xs: 4, sm: 8 }, paddingY: 4 }}>
        <PlayerInfo name={player.Name} team={player.TeamTrackmanAbbreviation} />
        {/* Nested routes render here: stats, heat-map, percentiles */}
        <Outlet />
      </Box>
    </Box>
  );
}
