import time

import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from .SessionState import session_get
from ..data import REDDIT_DATASET, REDDIT_META
from .util import timer

@st.cache
def read_csv_cached(path):
    return pd.read_csv(path)


@timer
def app():
    print()
    st.write(f"<style>{open('src/frontend/css/reddit_dataset.css').read()}</style>", unsafe_allow_html=True)

    @st.cache(show_spinner=False)
    def stats(df):
        return df.shape[0], df['user'].nunique(), df['subreddit'].nunique()

    t1 = time.time()

    subreddit_info = read_csv_cached(REDDIT_META)[['subreddit', 'num_subscribers', 'over18', 'public_description']]

    t2 = time.time()
    print("{0:<20}{1:.3f}s".format("DATASET:", t2 - t1))

    state = session_get(u_comments=20, u_reddit=20, r_comments=100, r_users=100, include_over18=False)

    @st.cache
    def group_subreddit(raw_dataset):
        print("Group Reddit")
        subreddit_group = raw_dataset.groupby(by=['subreddit'])['count']
        sum_comments_per_subreddit = subreddit_group.sum().reset_index(name="total_num_comments")
        unique_users_per_subreddit = subreddit_group.count().reset_index(name="unique_users")
        return sum_comments_per_subreddit.merge(unique_users_per_subreddit, on="subreddit")

    @st.cache
    def group_user(raw_dataset):
        print("Group User")
        user_group = raw_dataset.groupby(by=['user'])['count']
        sum_comments_per_user = user_group.sum().reset_index(name="total_num_comments")
        unique_subreddits_per_user = user_group.count().reset_index(name="unique_subreddits")
        return sum_comments_per_user.merge(unique_subreddits_per_user, on="user")

    @st.cache
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
        print("Filter")
        raw_dataset = read_csv_cached(REDDIT_DATASET)
        user_grouped = group_user(raw_dataset)
        subreddit_grouped = group_subreddit(raw_dataset)

        this_user = user_grouped[(user_grouped.unique_subreddits >= u_reddit)
                                 & (user_grouped.total_num_comments >= u_comments)]

        this_subreddit = subreddit_grouped[(subreddit_grouped.unique_users >= r_users)
                                           & (subreddit_grouped.total_num_comments >= r_comments)]

        filtered_df = raw_dataset[(raw_dataset.user.isin(this_user.user)) &
                                  (raw_dataset.subreddit.isin(this_subreddit.subreddit))]

        # remove all subreddits over 18
        if not include_over18:
            subreddits_under18 = filtered_df.subreddit.isin(subreddit_info[~subreddit_info.over18].subreddit)
            filtered_df = filtered_df[subreddits_under18]

        return filtered_df, user_grouped, subreddit_grouped

    @st.cache
    def group_subreddit_plot(df, limit=15, sort="by_user"):
        sort_by_col = "unique_users" if sort == "by user" else "total_num_comments"
        subreddit_summary = df.sort_values(by=sort_by_col, ascending=False, ignore_index=True)
        subreddit_summary = subreddit_summary.iloc[:limit]

        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           height=400, width=400, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis=dict(title="Total number of comments"),
                           yaxis2={'title': 'Number unique users', 'overlaying': 'y', 'side': 'right'})
        fig = go.Figure(layout=layout)
        fig.add_trace(go.Bar(x=subreddit_summary['subreddit'], opacity=1, name="Number of comments",
                             y=subreddit_summary['total_num_comments'], orientation='v', offsetgroup=1,
                             yaxis='y'))
        fig.add_trace(go.Bar(x=subreddit_summary['subreddit'], opacity=1, name="Unique users",
                             y=subreddit_summary['unique_users'], orientation='v', offsetgroup=2, yaxis='y2'))
        return fig

    @st.cache
    def group_user_plot(df, limit=15, sort="total_num_comments"):
        sort_by_col = "unique_subreddits" if sort == "by subreddits" else "total_num_comments"
        user_summary = df.sort_values(by=sort_by_col, ascending=False, ignore_index=True)
        user_summary = user_summary.iloc[:limit]

        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           height=400, width=400, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis=dict(title="Total number of comments"),
                           yaxis2={'title': 'Number unique subreddits', 'overlaying': 'y', 'side': 'right'})
        fig = go.Figure(layout=layout)
        fig.add_trace(go.Bar(x=user_summary['user'], opacity=1, name="Number of comments",
                             y=user_summary['total_num_comments'], orientation='v', offsetgroup=1,
                             yaxis='y'))
        fig.add_trace(go.Bar(x=user_summary['user'], opacity=1, name="Unique subreddits",
                             y=user_summary['unique_subreddits'], orientation='v', offsetgroup=2, yaxis='y2'))
        return fig

    @st.cache
    def subreddit_details(subreddit):
        return subreddit_info[subreddit_info['subreddit'] == subreddit].to_numpy()[0]

    t3 = time.time()
    # Sidebar
    st.sidebar.title("Configuration")
    filter_conf = st.sidebar.form(key="filter_conf")
    s_r_users = filter_conf.slider("Min. User per Subreddit",
                                   value=state.r_users, min_value=0, max_value=200, step=10,
                                   help="The minimum number of unique users a subreddit must have to be included in the "
                                        "dataset.")
    s_r_comments = filter_conf.slider("Min. Comment per Subreddit",
                                      value=state.r_comments, min_value=0, max_value=200, step=10,
                                      help="The minimum number of comments a subreddit must have to be included in the "
                                           "dataset.")
    s_u_comments = filter_conf.slider("Min. Comment per User",
                                      value=state.u_comments, min_value=0, max_value=200, step=10,
                                      help="The minimum number total number of comments a user must have written,")
    s_u_reddit = filter_conf.slider("Min. Subreddit per User",
                                    value=state.u_reddit, min_value=0, max_value=200, step=10,
                                    help="The number of different subreddits that a user must have commented on to be "
                                         "included in the dataset.")

    s_include_over18 = filter_conf.checkbox("Include subreddits over 18?", value=state.include_over18,
                                            help="Should subreddits be included that are not approved for minors?")

    if filter_conf.form_submit_button("Apply"):
        state.r_users = s_r_users
        state.r_comments = s_r_comments
        state.u_comments = s_u_comments
        state.u_reddit = s_u_reddit
        state.include_over18 = s_include_over18

    same = (state.r_users == s_r_users) & (state.r_comments == s_r_comments) & (
            state.u_comments == s_u_comments) & (state.u_reddit == s_u_reddit) & (
                   state.include_over18 == s_include_over18)

    t4 = time.time()
    print("{0:<20}{1:.3f}s".format("SIDEBAR:", t4 - t3))

    data, g_user, g_subreddit = filter_dataset(r_users=state.r_users, r_comments=state.r_comments,
                                               u_comments=state.u_comments, u_reddit=state.u_reddit,
                                               include_over18=state.include_over18)

    st.sidebar.text_input("Path", value="data.csv")
    if st.sidebar.button("Save"):
        data.to_csv("data.csv", index=False)

    t5 = time.time()
    print("{0:<20}{1:.3f}s".format("FILTER:", t5 - t4))
    data_size, num_users, num_reddits = stats(data)
    st.title("Subreddit Dataset")

    st.markdown("Information about Reddit ...")

    st.markdown("""<div class="content-container">
    <div class="content-wrapper">
            <div class="label">Total data points</div>
        <div class="value">{0:,}</div>
    </div>
    <div class="content-wrapper">
            <div class="label">Unique users</div>
        <div class="value">{1:,}</div>
    </div>
    <div class="content-wrapper">
            <div class="label">Unique subreddits</div>
        <div class="value">{2:,}</div>
    </div>
    </div>""".format(data_size, num_users, num_reddits), unsafe_allow_html=True)

    st.subheader("Histogram of Ratings")
    hist_col1, hist_col2 = st.beta_columns([1, 2])

    hist_col1.write(data['count'].describe())

    @st.cache
    def histogram():
        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           height=250, width=400, margin=dict(l=10, r=10, t=10, b=10))
        return go.Figure(data=go.Histogram(x=data['count'], nbinsx=50), layout=layout)

    t5 = time.time()
    hist_col2.plotly_chart(histogram(), use_container_width=True)
    t6 = time.time()
    print("{0:<20}{1:.3f}s".format("HISTOGRAM:", t6 - t5))

    c1, c2, c3 = st.beta_columns([2, 1, 1])
    c1.subheader("Top Subreddits")
    n_top_subreddits = c2.number_input("Number of subreddits", value=15, min_value=2, max_value=100, step=1)
    sort_column_subreddit = c3.radio("Sort", options=["by comments", "by user"], key="sort_reddit",
                                     help="Should the top users be displayed sorted by the number of comments or the "
                                          "number of unique subreddits they commented on.")
    st.plotly_chart(group_subreddit_plot(g_subreddit, limit=n_top_subreddits, sort=sort_column_subreddit),
                    use_container_width=True)

    c1, c2, c3 = st.beta_columns([2, 1, 1])
    c1.subheader("Top User")
    n_top_users = c2.number_input("Number of users", value=15, min_value=2, max_value=100, step=1)
    sort_column_user = c3.radio("Sort", options=["by comments", "by subreddits"],
                                help="Should the top users be displayed sorted by the number of comments or the number "
                                     "of unique subreddits they commented on.")
    st.plotly_chart(group_user_plot(g_user, limit=n_top_users, sort=sort_column_user), use_container_width=True)
    t7 = time.time()
    print("{0:<20}{1:.3f}s".format("CHARTS:", t7 - t6))

    st.subheader("Subreddit and User details")
    with st.beta_expander("Subreddit details"):
        detail_subreddit = st.selectbox("Subreddit", options=data['subreddit'].unique())
        _, num_subscribers, over18, public_description = subreddit_details(subreddit=detail_subreddit)
        st.subheader("Description")
        st.markdown(public_description)
        st.markdown("{0:,} Subscribers".format(num_subscribers))
        st.markdown("{0}".format("🔞" if over18 else "No age restriction."))
        st.markdown(f"[Link to forum](https://www.reddit.com/r/{detail_subreddit}/)")

    with st.beta_expander("User details"):
        detail_user = st.selectbox("User", options=data['user'].unique())
        st.write(data[data['user'] == detail_user][['subreddit', 'count']])
    t8 = time.time()
    print("{0:<20}{1:.3f}s".format("DETAILS:", t8 - t7))
    print("-" * 26)
