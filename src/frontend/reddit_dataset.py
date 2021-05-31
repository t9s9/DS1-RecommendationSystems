import time

import pandas as pd
import plotly.graph_objs as go
import streamlit as st

import SessionState

save_path = "C:/Users/TS/PycharmProjects/DS1-RecommendationSystems/data/reddit/"


def app():
    @st.cache
    def read_csv_cached(path):
        return pd.read_csv(path)

    @st.cache(show_spinner=False)
    def stats(df):
        return df.shape[0], df['user'].nunique(), df['subreddit'].nunique()

    t1 = time.time()
    raw_dataset = read_csv_cached(save_path + "dataset.csv")
    raw_user_summary = read_csv_cached(save_path + "user_summary.csv")
    raw_subreddit_summary = read_csv_cached(save_path + "subreddit_summary.csv")
    # subreddit_info = read_csv_cached(save_path + "subreddit_info.csv")

    print("Loading time: {:3f}s".format(time.time() - t1))

    data_state = SessionState.get(u_comments=10, u_reddit=10, r_comments=10, r_users=10, include_over18=True)

    @st.cache(show_spinner=False)
    def filter_dataset(u_comments, u_reddit, r_comments, r_users, include_over18):
        """

        :param u_comments: The minimum total number of comments a user should have.
        :param u_reddit: The minimum number of unique subreddits a user must have commented on.
        :param r_comments: The minimum number of comments a subreddit must have.
        :param r_users: The number of unique users a subreddit must have.
        :param include_over18:
        :return:
        """
        # filter user
        this_users = raw_user_summary[(raw_user_summary.unique_subreddits >= u_reddit)
                                      & (raw_user_summary.total_num_comments >= u_comments)]
        progress.progress(20)
        this_subreddit = raw_subreddit_summary[
            (raw_subreddit_summary.unique_users >= r_users)
            & (raw_subreddit_summary.total_num_comments >= r_comments)]
        progress.progress(40)
        q = raw_dataset[(raw_dataset.user.isin(this_users.user)) &
                        (raw_dataset.subreddit.isin(this_subreddit.subreddit))]
        progress.progress(80)
        # remove all subreddits over 18
        if not include_over18:
            subreddits_under18 = q.subreddit.isin(subreddit_info[~subreddit_info.over18].subreddit)
            q = q[subreddits_under18]
        progress.progress(100)
        return q

    # Sidebar
    title_container = st.sidebar.empty()
    s_r_users = st.sidebar.slider("Min. User per Subreddit",
                                  value=data_state.r_users, min_value=0, max_value=1000,
                                  help="The minimum number of comments a subreddit must have to be included in the dataset.")
    s_r_comments = st.sidebar.slider("Min. Comment per Subreddit",
                                     value=data_state.r_comments, min_value=0, max_value=1000,
                                     help="The minimum number of comments a subreddit must have to be included in the dataset.")

    s_u_comments = st.sidebar.slider("Min. Comment per User",
                                     value=data_state.u_comments, min_value=0, max_value=1000,
                                     help="The number of different subreddits that a user must have commented on to be included in the "
                                          "dataset.")
    s_u_reddit = st.sidebar.slider("Min. Subreddit per User",
                                   value=data_state.u_reddit, min_value=0, max_value=1000,
                                   help="The number of different subreddits that a user must have commented on to be included in the "
                                        "dataset.")

    s_include_over18 = st.sidebar.checkbox("Include subreddits over 18?", value=data_state.include_over18,
                                           help="Should subreddits be included that are not approved for minors?")
    progress = st.sidebar.progress(0)

    if st.sidebar.button("Apply"):
        data_state.r_users = s_r_users
        data_state.r_comments = s_r_comments
        data_state.u_comments = s_u_comments
        data_state.u_reddit = s_u_reddit
        data_state.include_over18 = s_include_over18

    same = (data_state.r_users == s_r_users) & (data_state.r_comments == s_r_comments) & (
            data_state.u_comments == s_u_comments) & (data_state.u_reddit == s_u_reddit) & (
                   data_state.include_over18 == s_include_over18)

    title_container.title("Configuration {0}".format("🟢" if same else "🔴"))

    data = filter_dataset(data_state.r_users, data_state.r_comments,
                          data_state.u_comments,
                          data_state.u_reddit, data_state.include_over18)

    st.title("Subreddit Recommender")
    st.text("Information about Reddit ...")
    data_size, num_users, num_reddits = stats(data)
    st.text("{0:<20}{1:,}".format("Total datapoints:", data_size))
    st.text("{0:<20}{1:,}".format("Unique users:", num_users))
    st.text("{0:<20}{1:,}".format("Unique reddits:", num_reddits))

    graph_col1, graph_col2 = st.beta_columns([1, 1])
    graph_col1.subheader("Top 15 Subreddits")
    layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       height=600, width=400, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(title="Total number of comments"),
                       xaxis2={'title': 'Number unique users', 'overlaying': 'x', 'side': 'top'})
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Bar(x=raw_subreddit_summary.iloc[:15]['total_num_comments'], opacity=1, name="Number of comments",
                         y=raw_subreddit_summary.iloc[:15]['subreddit'], orientation='h', offsetgroup=1, xaxis='x'))

    fig.add_trace(go.Bar(x=raw_subreddit_summary.iloc[:15]['unique_users'], opacity=1, name="Unique users",
                         y=raw_subreddit_summary.iloc[:15]['subreddit'], orientation='h', offsetgroup=2, xaxis='x2'))
    fig.update_layout(barmode='group')
    graph_col1.plotly_chart(fig, use_container_width=True)

    # col1, col2 = st.beta_columns([1, 1])
    # col1.subheader("Top 20 subreddits")
    #
    # col2.subheader("Top 20 user")
    #
    # detail_col1, detail_col2 = st.beta_columns([1, 1])
    # with detail_col1:
    #     with st.beta_expander("Subreddit details"):
    #         detail_subreddit = st.selectbox("Subreddit", options=data['subreddit'].unique())
    #         num_subscribers, over18, public_description, details = subreddit_details(subreddit=detail_subreddit, db=db)
    #         st.subheader("Description")
    #         st.markdown(public_description)
    #         st.markdown("{0:,} Subscribers".format(num_subscribers))
    #         st.markdown("{0}".format("Over 18" if over18 else "No age restriction"))
    #         st.subheader("Details")
    #         st.markdown(details, unsafe_allow_html=True)
    #
    # with detail_col2:
    #     with st.beta_expander("User details"):
    #         detail_user = st.selectbox("User", options=data['user'].unique())
    #         st.write(data[data['user'] == detail_user][['subreddit', 'count']])
