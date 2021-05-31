use reqwest;
use serde::{Serialize,Deserialize};
use serde_json;
use rusqlite::{params, Connection, Result};

#[derive(Serialize,Deserialize)]
struct Summoner {
    #[serde(rename = "accountId")]
    account_id: String,
    id: String,
}

#[derive(Serialize,Deserialize)]
struct Matches {
    #[serde(rename="gameId")]
    game_id: usize,
}

#[derive(Serialize,Deserialize)]
struct GameList {
    matches: Vec<Matches>,
}

#[derive(Serialize,Deserialize)]
struct Match {
    #[serde(rename="participantIdentities")]
    participant_ids: Vec<ParticipantIdentities>,
    #[serde(rename="gameType")]
    game_type: String,
    participants: Vec<Participant>,
}

#[derive(Serialize,Deserialize)]
struct Participant {
    #[serde(rename="championId")]
    champion_id: usize,
    stats: Stats,
}

#[derive(Serialize,Deserialize)]
struct Stats {
    win: bool,
}

#[derive(Serialize,Deserialize)]
struct ParticipantIdentities {
    player: Player,
}

#[derive(Serialize,Deserialize)]
struct Player {
    #[serde(rename="accountId")]
    account_id: String,
    #[serde(rename="summonerName")]
    summoner_name: String,
}


struct ApiServer {
    crawled_summoners: std::collections::HashSet<String>,
    canidate_summoners: std::collections::HashSet<String>,
    crawled_games: std::collections::HashSet<usize>,
    server: String,
}

struct ApiCrawler {
    sqlite: std::sync::Arc<std::sync::Mutex<Connection>>,
    keys: Vec<String>,
    length: usize,
    servers: std::sync::Arc<Vec<std::sync::RwLock<ApiServer>>>,
}

impl ApiCrawler {
    pub fn run(&self) {
        let stop_cond = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(self.length as u32));
        let mut threads = Vec::new();
        let other = stop_cond.clone();
        ctrlc::set_handler(move || {
            other.store(true, std::sync::atomic::Ordering::SeqCst);
        }).expect("Error setting Ctrl-C handler");
        for x in self.keys.iter() {
            for z in 0..self.servers.len() {
                let moved = x.to_string();
                let arc = self.servers.clone();
                let at = stop_cond.clone();
                let sql = self.sqlite.clone();
                let mcounter = counter.clone();
                threads.push(std::thread::spawn(move ||start_crawling(z,moved, at, arc,sql,mcounter)));
            }
        }

        threads.into_iter().for_each(|x|x.join().unwrap());
    }

    pub fn new(conn: Connection, server: Vec<&'static str>,keys: Vec<&str>) -> ApiCrawler {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS matches (id INTEGER,desc TEXT,idx TEXT);",
            [],
        ).unwrap();
        conn.execute(
            "CREATE TABLE IF NOT EXISTS crawled_summoners (name TEXT,idx TEXT);",
            [],
        ).unwrap();
        conn.execute(
            "CREATE TABLE IF NOT EXISTS canidate_summoners (name TEXT,idx TEXT);",
            [],
        ).unwrap();
    
        let mut vec = Vec::new();
        let mut length = 0;
        for x in server.iter() {
            let mut stmt = conn.prepare("SELECT id from matches where idx=?").unwrap();
            let crawled_games = stmt.query_map([x], |row| {
                Ok(row.get(0).unwrap())
            }).unwrap().collect::<Result<std::collections::HashSet<usize>,_>>().unwrap();

            stmt = conn.prepare("SELECT name from crawled_summoners where idx=?").unwrap();
            let crawled_summoners = stmt.query_map([x], |row| {
                Ok(row.get(0).unwrap())
            }).unwrap().collect::<Result<std::collections::HashSet<String>,_>>().unwrap();

            stmt = conn.prepare("SELECT name from canidate_summoners where idx=?").unwrap();
            let mut canidate_summoners = stmt.query_map([x], |row| {
                Ok(row.get(0).unwrap())
            }).unwrap().collect::<Result<std::collections::HashSet<String>,_>>().unwrap();
            if canidate_summoners.len() == 0 {
                canidate_summoners.insert("shadow".to_string());
                canidate_summoners.insert("apple".to_string());
            }

            length += crawled_games.len();
            vec.push(std::sync::RwLock::new(ApiServer {
                crawled_summoners,
                canidate_summoners,
                crawled_games,
                server: x.to_string(),
            }));
        }

        ApiCrawler {
            sqlite: std::sync::Arc::new(std::sync::Mutex::new(conn)),
            servers: std::sync::Arc::new(vec),
            length: length,
            keys: keys.iter().map(|x|x.to_string()).collect(),
        }
    }
}

fn check_timestamp(time: &mut Vec<std::time::Instant>) {
    time.push(std::time::Instant::now());
    while time.iter().filter(|x| x.elapsed().as_millis() < 1300).count() >= 20 {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    while time.iter().filter(|x| x.elapsed().as_secs() < 125).count() >= 100 {
        std::thread::sleep(std::time::Duration::from_millis(1000));
    }

    time.retain(|x| x.elapsed().as_secs() < 125);
}

fn start_crawling(base_path: usize, api_key: String, stop_cond: std::sync::Arc<std::sync::atomic::AtomicBool>, api: std::sync::Arc<Vec<std::sync::RwLock<ApiServer>>>,sqlite: std::sync::Arc<std::sync::Mutex<Connection>>,counter: std::sync::Arc<std::sync::atomic::AtomicU32>) {
    let mut time_vec = Vec::new();
    let api_base = api[base_path].read().unwrap().server.clone();

    while !stop_cond.load(std::sync::atomic::Ordering::Relaxed) {
        let current_summoner = {
            let mut access = api[base_path].write().unwrap();
            let current_summoner = if let Some(summoner) = access.canidate_summoners.iter().next() {
                summoner.clone()
            } else { panic!("This should never happen!"); };
            access.canidate_summoners.remove(&current_summoner);
            let sql = sqlite.lock().unwrap();
            sql.execute(
                         "DELETE FROM canidate_summoners where name=? AND idx=?",
                            params![current_summoner,api_base],
                        ).unwrap();
            current_summoner
        };
        let summoner_url = format!("{}/lol/summoner/v4/summoners/by-name/{}?api_key={}",&api_base,&current_summoner,api_key);
        check_timestamp(&mut time_vec);
        let summoner_response = reqwest::blocking::get(summoner_url).unwrap();
        if summoner_response.status().as_u16() != 200 {
            if summoner_response.status().as_u16() == 403 {
                println!("It seems that the API key has expired, the server will gracefully shut down now!");
                stop_cond.store(true,std::sync::atomic::Ordering::SeqCst);
                break;
            }
            println!("Got error {} while fetching summoner {}, skipping summoner...",summoner_response.status().as_u16(),current_summoner);
            continue;
        }
        let summoner: Summoner = summoner_response.json().unwrap();

        let game_url = format!("{}/lol/match/v4/matchlists/by-account/{}?api_key={}",&api_base,&summoner.account_id,&api_key);
        {
            let mut access = api[base_path].write().unwrap();

            if access.crawled_summoners.get(&current_summoner).is_none() {
                access.crawled_summoners.insert(current_summoner.clone());
                let sql = sqlite.lock().unwrap();

                sql.execute(
                         "INSERT INTO crawled_summoners (name,idx) VALUES (?1,?2)",
                            params![current_summoner,api_base],
                        ).unwrap();
            }
        }
        check_timestamp(&mut time_vec);
        let games_resp: reqwest::blocking::Response = reqwest::blocking::get(game_url).unwrap();
        if games_resp.status().as_u16() != 200 {
            if games_resp.status().as_u16() == 403 {
                println!("It seems that the API key has expired, the server will gracefully shut down now!");
               stop_cond.store(true,std::sync::atomic::Ordering::SeqCst);
               break;
            }
            println!("Got error {} while fetching matches of summoner {}, skipping games...",games_resp.status().as_u16(),current_summoner);
            continue;
        }
        let games: GameList = games_resp.json().unwrap();
        for x in games.matches.into_iter() {
            if stop_cond.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            let result = {
                let access = api[base_path].read().unwrap();
                access.crawled_games.get(&x.game_id).is_none()
            };

            if result {
                let match_url = format!("{}/lol/match/v4/matches/{}?api_key={}",&api_base, x.game_id,&api_key);
                check_timestamp(&mut time_vec);
                let match_response  = reqwest::blocking::get(match_url).unwrap();
                if match_response.status().as_u16() != 200 {
                    println!("Got error {} while fetching match {} of summoner {}, skipping match...",match_response.status().as_u16(),x.game_id,current_summoner);
                    if match_response.status().as_u16() == 403 {
                        println!("It seems that the API key has expired, the server will gracefully shut down now!");
                        stop_cond.store(true,std::sync::atomic::Ordering::SeqCst);
                        break;
                    }
                    continue;
                }
                let single_match: Match = match_response.json().unwrap();
                if single_match.game_type == "MATCHED_GAME" {
                    let mut access = api[base_path].write().unwrap();
                    let sql = sqlite.lock().unwrap();
                    for x in single_match.participant_ids.iter() {
                        access.canidate_summoners.insert(x.player.summoner_name.clone());
                        sql.execute(
                         "INSERT INTO canidate_summoners (name,idx) VALUES (?1,?2)",
                            params![x.player.summoner_name,api_base],
                        ).unwrap();
                    }
                    access.crawled_games.insert(x.game_id);
                    sql.execute(
                         "INSERT INTO matches (id,desc,idx) VALUES (?1,?2,?3)",
                            params![x.game_id,serde_json::to_string(&single_match.participants).unwrap(),api_base],
                        ).unwrap();
                    counter.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
                }
                println!("Crawled a total of {} games",counter.load(std::sync::atomic::Ordering::Relaxed));
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conn = Connection::open("league_of_legends.db").unwrap();
    let api_crawler = ApiCrawler::new(conn,vec!["https://euw1.api.riotgames.com","https://eun1.api.riotgames.com","https://na1.api.riotgames.com","https://ru.api.riotgames.com","https://la1.api.riotgames.com","https://la2.api.riotgames.com"],
vec!["RGAPI-3b27b3fa-05aa-455a-a6cb-b90bce70afc7", "RGAPI-7c7f6b92-23d4-4759-937f-2dd521a4a299"]);
    api_crawler.run();
    Ok(())
}
